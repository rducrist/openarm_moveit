"""The OpenArm cup_table policy's deployment interface: raw robot data in, TCP targets out (policy environment).

Loads one policy file written by export_policy.py, which carries the weights, the training config, the action
statistics, the camera calibration and the data generation settings, and reproduces the training data path exactly
(tinydp.dataset.image_dataset_sim + tinydp.policy.model), so robot code only deals with physical quantities:

Observation: obs_horizon control steps (frame_dt_s apart, the data's log_hz), oldest first:
  frames  the robot camera as captured (BGR, raw_size: e.g. the 1920x1080 stream); undistorted here with the bundled
          calibration (K kept) and cropped to the policy's 1280x720 window when the calibration is a crop (the sim
          renders a pinhole camera with the crop's K), then tinydp's default_vision_transform:
          resize to image_size x image_size (no crop, aspect squashed), ImageNet mean/std
  TCPs    per arm: openarm_{left,right}_hand_tcp in the robot base frame (URDF `world`), position [m] + quaternion wxyz
  gripper per arm: the last gripper command, 1 = closed, 0 = open
Action chunk: num_pred_steps steps, step_dt_s apart, starting one step after the newest observation, for the policy's
  arm (`arm`, pred_left_only / pred_right_only; the other arm is held at its last pose): TCP position [m] + quaternion
  wxyz in the same frame, and a gripper command (> 0.5: close). Execute part of the chunk, then predict again.
  Denoising: DDIM, 10 steps by default (~60 ms on an RTX 5090; training's DDPM over 100 steps: ~340 ms).
policy.spec() / the .json next to the policy file give the same, with the numbers.

    pixi run -e policy python scripts/pixi/openarm_policy.py <policy.pt> \\
        --rollout output_data/<task>/successes/<env_...> --image scans/_incoming_robotcam/<robot photo>.jpg

checks the file end to end on a simulated rollout (vs its recorded future) and on a real robot camera frame.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Sequence

import cv2
import h5py
import numpy as np
import torch
import yaml
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from PIL import Image
from tinydp.dataset.utils import default_vision_transform, quat_to_rot_6d, rot_6d_to_quat, unscale_action
from tinydp.policy.model import DiffusionPolicy

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tinydp_vision  # noqa: E402

FORMAT = "openarm_cup_policy/1"


class OpenArmCupPolicy:
    def __init__(self, policy_file: Path, device: str = "cuda", inference_steps: int | None = 10):
        """inference_steps: DDIM denoising steps, as tinydp's DiffusionWrapper deploys (10: ~10x faster than training's
        DDPM, same weights); None: the model's DDPM over all num_train_diffusion_steps."""
        bundle = torch.load(policy_file, map_location="cpu", weights_only=False)
        if bundle.get("format") != FORMAT:
            raise ValueError(f"{policy_file}: not an {FORMAT} file (made by scripts/pixi/export_policy.py)")
        run = yaml.load(bundle["run_yaml"], Loader=yaml.Loader)
        # the backbone it was trained with (bundles before tinydp_vision.py: tinydp's own); weights from the file
        tinydp_vision.apply(bundle.get("vision_arch", {"backbone": "resnet18_avgpool"}), pretrained=False)
        self.model = DiffusionPolicy(model_cfg=run.model_cfg, shared_cfg=run.shared_cfg)
        self.model.load_state_dict(bundle["model"])
        self.model = self.model.to(device).eval()
        if inference_steps:
            ddpm = self.model.noise_scheduler.config
            self.model.noise_scheduler = DDIMScheduler(num_train_timesteps=ddpm.num_train_timesteps, beta_schedule=ddpm.beta_schedule,
                                                       clip_sample=ddpm.clip_sample, prediction_type=ddpm.prediction_type)
            self.model.noise_scheduler.set_timesteps(inference_steps)
        self.inference_steps = inference_steps or self.model.noise_scheduler.config.num_train_timesteps
        stats = bundle["action_statistics"]
        shape = stats["shape"]
        self.stats = {"min": torch.tensor(np.array(stats["min_action"]).reshape(shape)),
                      "max": torch.tensor(np.array(stats["max_action"]).reshape(shape))}
        self.bundle_info = {k: bundle[k] for k in ("interface", "dataset_info", "provenance", "weights")}
        self.bundle_info["vision_arch"] = bundle.get("vision_arch", {"backbone": "resnet18_avgpool"})
        cal = bundle["calibration"]
        self.K, self.dist, self.size = np.array(cal["k"]), np.array(cal["dist"]), (cal["w"], cal["h"])
        # a crop of a larger stream (make_crop_calibration.py): raw frames are the source size, undistorted with the
        # source K, then cropped; otherwise raw frames are the calibrated size itself (older calibrations have a
        # note string as `source`)
        src = cal.get("source")
        src = src if isinstance(src, dict) and "crop_xywh" in src else None
        self.raw_size = (src["w"], src["h"]) if src else self.size
        self.K_raw = np.array(src["k"]) if src else self.K
        self.crop = src["crop_xywh"] if src else None
        self.device = device
        self.obs_horizon = run.shared_cfg.seq_length
        self.horizon = run.shared_cfg.num_pred_steps
        self.image_size = run.shared_cfg.image_size
        self.frame_dt = bundle["interface"]["observation"]["frame_dt_s"]
        self.action_dt = bundle["interface"]["action"]["step_dt_s"]
        self.subsample = run.dataset_cfg.subsample_steps
        self.arm = "right" if run.model_cfg.policy_cfg.pred_right_only else ("left" if run.model_cfg.policy_cfg.pred_left_only else "both")
        if self.arm == "both":
            raise ValueError(f"{policy_file}: bimanual policies are not handled by this interface yet")
        # as in training (tinydp_vision.apply_crop): the policy's crop of the camera image, then tinydp's transform
        self.image_crop = self.bundle_info["vision_arch"].get("image_crop")
        self.transform = tinydp_vision.with_crop(default_vision_transform(resolution=self.image_size), self.image_crop)
        if run.shared_cfg.num_cameras != 1:
            raise ValueError(f"{policy_file}: trained with {run.shared_cfg.num_cameras} cameras, this interface feeds the robot camera only")

    def spec(self) -> dict:
        return self.bundle_info

    def preprocess(self, frame: np.ndarray, bgr: bool = True, raw: bool = True) -> torch.Tensor:
        """frame: the raw camera frame (raw_size, distorted, BGR); raw=False: an already undistorted, cropped frame
        (e.g. a simulated one), of the policy's size."""
        if raw:
            if (frame.shape[1], frame.shape[0]) != self.raw_size:
                raise ValueError(f"frame is {frame.shape[1]}x{frame.shape[0]}, the raw camera frames are {self.raw_size}")
            frame = cv2.undistort(frame, self.K_raw, self.dist, None, self.K_raw)
            if self.crop:
                x0, y0, w, h = self.crop
                frame = frame[y0:y0 + h, x0:x0 + w]
        elif (frame.shape[1], frame.shape[0]) != self.size:
            raise ValueError(f"frame is {frame.shape[1]}x{frame.shape[0]}, the policy's images are {self.size}")
        if bgr:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.transform(Image.fromarray(frame))  # as tinydp's dataset: PIL RGB image -> transform

    @staticmethod
    def arm_state(tcp_xyz_wxyz: np.ndarray, gripper_closed: float) -> np.ndarray:
        tcp = np.asarray(tcp_xyz_wxyz, dtype=np.float64)
        return np.concatenate([tcp[:3], quat_to_rot_6d(tcp[None, 3:])[0], [float(gripper_closed)]])

    @torch.no_grad()
    def predict(self, frames: Sequence[torch.Tensor], left_tcp: np.ndarray, right_tcp: np.ndarray,
                grippers: np.ndarray) -> dict:
        """frames: obs_horizon preprocess() outputs; left/right_tcp: (obs_horizon, 7) xyz + wxyz; grippers: (obs_horizon, 2)."""
        T = self.obs_horizon
        if not (len(frames) == T and len(left_tcp) == T and len(right_tcp) == T and len(grippers) == T):
            raise ValueError(f"need {T} observations (frame_dt {self.frame_dt:.4f} s apart), oldest first")
        obs = torch.stack(list(frames))[None, :, None].to(self.device)  # 1, T, cameras, 3, S, S
        proprio = np.stack([np.concatenate([self.arm_state(l, g[0]), self.arm_state(r, g[1])])
                            for l, r, g in zip(left_tcp, right_tcp, grippers)])
        proprio = torch.tensor(proprio[None], dtype=torch.float32, device=self.device)  # 1, T, 20
        naction = self.model.forward_inference({"observation": obs, "proprio": proprio}, vision_transform=None)
        # one arm predicted: unscale with the 20-d statistics, as tinydp's DiffusionWrapper does
        naction = torch.cat([naction, naction], dim=-1)
        action = unscale_action(naction, stat=self.stats, type="diffusion")[0].cpu().numpy().astype(np.float64)
        a = action[:, :10] if self.arm == "left" else action[:, 10:]
        held = left_tcp if self.arm == "right" else right_tcp
        return {"arm": self.arm, "tcp_xyz": a[:, :3], "tcp_wxyz": rot_6d_to_quat(a[:, 3:9]), "gripper": a[:, 9],
                "other_arm_tcp": np.asarray(held[-1]), "step_dt_s": self.action_dt}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("policy_file", type=Path)
    parser.add_argument("--rollout", type=Path, help="a sim rollout: predict at a few times, compare with its future TCPs")
    parser.add_argument("--image", type=Path, help="a raw robot camera frame, with the data's home TCP poses as proprio")
    parser.add_argument("--inference-steps", type=int, default=10, help="DDIM steps; 0: training's DDPM (all steps)")
    args = parser.parse_args()

    policy = OpenArmCupPolicy(args.policy_file, inference_steps=args.inference_steps or None)
    prov = policy.spec()["provenance"]
    print(f"{args.policy_file}: commit {prov['git_commit'][:8]}{' (dirty)' if prov['git_dirty'] else ''}, "
          f"{prov['training_rollouts']} rollouts, {policy.spec()['weights']} weights; obs {policy.obs_horizon} x "
          f"{policy.frame_dt * 1000:.1f} ms at {policy.image_size}px, actions {policy.horizon} x {policy.action_dt * 1000:.1f} ms, "
          f"{policy.inference_steps} denoising steps")
    T = policy.obs_horizon
    if args.rollout:
        f = h5py.File(args.rollout / "robot_data" / "robot_data.h5", "r")
        ee, grip = f["ee_poses"][:], f["gripper_binary_cmd"][:].astype(float)
        for t in (T, len(ee) // 3, 2 * len(ee) // 3):
            idx = list(range(t - T + 1, t + 1))
            frames = [policy.preprocess(cv2.imread(str(args.rollout / "camera_0" / "rgb" / f"{i:04d}.jpg")), raw=False) for i in idx]
            t0 = time.time()
            out = policy.predict(frames, ee[idx, :7], ee[idx, 7:], grip[idx])
            ms = (time.time() - t0) * 1000
            future = [t + policy.subsample * (k + 1) for k in range(policy.horizon) if t + policy.subsample * (k + 1) < len(ee)]
            col = slice(0, 3) if policy.arm == "left" else slice(7, 10)
            err = np.linalg.norm(out["tcp_xyz"][:len(future)] - ee[future, col], axis=1)
            print(f"rollout frame {t}: {ms:.0f} ms; {policy.arm} TCP chunk {out['tcp_xyz'][0].round(3).tolist()} .. "
                  f"{out['tcp_xyz'][-1].round(3).tolist()}, gripper {out['gripper'][[0, -1]].round(2).tolist()}; "
                  f"vs the rollout's future: mean {err.mean() * 100:.1f} cm, max {err.max() * 100:.1f} cm")
    if args.image:
        frame = cv2.imread(str(args.image))
        if (frame.shape[1], frame.shape[0]) != policy.raw_size:  # interface check only: a frame of another video mode
            print(f"(resizing {args.image.name} from {frame.shape[1]}x{frame.shape[0]} to {policy.raw_size}: geometry not exact)")
            frame = cv2.resize(frame, policy.raw_size)
        # proprio: the first frame of --rollout (the robot at home), else both arms hanging
        if args.rollout:
            left, right = ee[0, :7], ee[0, 7:]
        else:
            left, right = np.array([0.0, 0.152, 0.703, 0.0, 1.0, 0.0, 0.0]), np.array([0.0, -0.152, 0.703, 0.0, 1.0, 0.0, 0.0])
        out = policy.predict([policy.preprocess(frame)] * T, np.repeat(left[None], T, 0), np.repeat(right[None], T, 0), np.zeros((T, 2)))
        print(f"real robot camera {args.image.name}: {policy.arm} TCP chunk {out['tcp_xyz'][0].round(3).tolist()} .. "
              f"{out['tcp_xyz'][-1].round(3).tolist()}, quat[0] {out['tcp_wxyz'][0].round(3).tolist()}, "
              f"gripper {out['gripper'].round(2).tolist()}")


if __name__ == "__main__":
    main()
