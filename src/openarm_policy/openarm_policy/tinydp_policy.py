"""Load and run a tinydp diffusion policy checkpoint (as trained by Real2Render2Real).

Checkpoint folder: run.yaml, action_statistics.json, checkpoint_<epoch>.pt.

Follows tinydp's DiffusionWrapper (tinydp/policy/diffusion_wrapper.py @ 76d8f9c) without its
problems: that file fails to import (tinydp.dataset.dataset does not exist), and it needs
`tinydp.util.args` to unpickle run.yaml. Here run.yaml is read as plain data and only
tinydp.policy.model is imported (third_party/tinydp must be on PYTHONPATH; the pixi `policy`
environment sets that up).

Model I/O (tinydp, R2R2R sim data):
  images   (T, S, S, 3) uint8 RGB, T = seq_length (oldest first), S = image_size; resized
           like training (PIL bilinear) before they get here
  proprio  (T, 20) [L xyz, L rot6d, L grip, R xyz, R rot6d, R grip], unnormalised
  output   (num_pred_steps, 20) absolute TCP poses + gripper, same layout. For a single-arm
           policy the other arm is filled with its newest proprio (as the wrapper does).
"""

import glob
import json
import os
import re
from types import SimpleNamespace

import numpy as np
import yaml

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class _RunYamlLoader(yaml.SafeLoader):
    """Reads run.yaml (a yaml.dump of tinydp dataclasses) without constructing objects."""


_RunYamlLoader.add_multi_constructor(
    "tag:yaml.org,2002:python/object:",
    lambda loader, suffix, node: loader.construct_mapping(node, deep=True))
_RunYamlLoader.add_multi_constructor(
    "tag:yaml.org,2002:python/object/apply:",
    lambda loader, suffix, node: (loader.construct_sequence(node, deep=True)
                                  if isinstance(node, yaml.SequenceNode)
                                  else loader.construct_mapping(node, deep=True)))
_RunYamlLoader.add_constructor(
    "tag:yaml.org,2002:python/tuple",
    lambda loader, node: tuple(loader.construct_sequence(node, deep=True)))


def load_run_config(ckpt_dir: str) -> dict:
    with open(os.path.join(ckpt_dir, "run.yaml"), encoding="utf-8") as f:
        return yaml.load(f, Loader=_RunYamlLoader)  # noqa: S506 - safe loader subclass


def _namespace(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: _namespace(v) for k, v in d.items()})
    return d


def latest_checkpoint(ckpt_dir: str) -> int:
    ids = [int(m.group(1)) for p in glob.glob(os.path.join(ckpt_dir, "checkpoint_*.pt"))
           if (m := re.search(r"checkpoint_(\d+)\.pt$", p))]
    if not ids:
        raise FileNotFoundError(f"no checkpoint_<N>.pt in {ckpt_dir}")
    return max(ids)


class TinyDPPolicy:
    def __init__(self, ckpt_dir: str, ckpt_id: int | None = None, weights: str = "model",
                 device: str = "cuda", ddim_steps: int = 0):
        import torch
        from tinydp.policy.model import DiffusionPolicy

        cfg = load_run_config(ckpt_dir)
        shared, policy_cfg = cfg["shared_cfg"], cfg["model_cfg"]["policy_cfg"]
        if shared.get("use_delta_action"):
            raise NotImplementedError("checkpoint was trained with delta actions; only "
                                      "absolute actions are supported")
        self.obs_horizon = int(shared["seq_length"])
        self.action_horizon = int(shared["num_pred_steps"])
        self.image_size = int(shared["image_size"])
        self.num_cameras = int(shared["num_cameras"])
        self.subsample_steps = int(cfg["dataset_cfg"].get("subsample_steps", 1))
        if policy_cfg["pred_left_only"]:
            self.arms = ["left"]
        elif policy_cfg["pred_right_only"]:
            self.arms = ["right"]
        else:
            self.arms = ["left", "right"]

        stats = json.load(open(os.path.join(ckpt_dir, "action_statistics.json")))
        shape = stats["shape"]
        self.action_min = np.asarray(stats["min_action"], dtype=np.float64).reshape(shape)
        self.action_max = np.asarray(stats["max_action"], dtype=np.float64).reshape(shape)

        if device.startswith("cuda") and not torch.cuda.is_available():
            print("CUDA not available, running the policy on the CPU (slow)")
            device = "cpu"
        self.device = torch.device(device)
        self.ckpt_id = latest_checkpoint(ckpt_dir) if ckpt_id is None else ckpt_id
        model = DiffusionPolicy(model_cfg=_namespace(cfg["model_cfg"]),
                                shared_cfg=_namespace(shared))
        ckpt = torch.load(os.path.join(ckpt_dir, f"checkpoint_{self.ckpt_id}.pt"),
                          map_location="cpu", weights_only=True)
        state = {k.removeprefix("module."): v for k, v in ckpt[weights].items()}
        model.load_state_dict(state)
        self.epoch = ckpt.get("epoch")
        del ckpt
        self.model = model.to(self.device).eval()

        if ddim_steps > 0:
            from diffusers.schedulers.scheduling_ddim import DDIMScheduler
            self.scheduler = DDIMScheduler(num_train_timesteps=model.num_diffusion_iters,
                                           beta_schedule="squaredcos_cap_v2", clip_sample=True,
                                           prediction_type=model.prediction_type)
            self.scheduler.set_timesteps(ddim_steps)
        else:  # what tinydp's wrapper runs: full DDPM chain of the training scheduler
            self.scheduler = model.noise_scheduler
            self.scheduler.set_timesteps(model.num_diffusion_iters)
        self._mean = torch.tensor(IMAGENET_MEAN, device=self.device).view(1, 3, 1, 1)
        self._std = torch.tensor(IMAGENET_STD, device=self.device).view(1, 3, 1, 1)

    def describe(self) -> str:
        return (f"tinydp checkpoint {self.ckpt_id} (epoch {self.epoch}): arms={self.arms}, "
                f"obs history {self.obs_horizon}, chunk {self.action_horizon}, "
                f"{self.num_cameras} camera(s) at {self.image_size}px, "
                f"{len(self.scheduler.timesteps)} denoising steps on {self.device}")

    def predict(self, images: np.ndarray, proprio: np.ndarray) -> np.ndarray:
        import torch
        T, S = self.obs_horizon, self.image_size
        assert images.shape == (T, S, S, 3), images.shape
        assert proprio.shape == (T, 20), proprio.shape
        with torch.inference_mode():
            x = torch.from_numpy(images).to(self.device).permute(0, 3, 1, 2).float() / 255.0
            x = (x - self._mean) / self._std
            nimage = x.view(1, T, 1, 3, S, S)
            nprop = torch.from_numpy(proprio.astype(np.float32)).to(self.device).unsqueeze(0)
            feats = self.model._get_observation_features(nimage, nprop)
            cond = feats.flatten(start_dim=1)
            a = torch.randn((1, self.action_horizon, self.model.action_dim), device=self.device)
            unet = self.model.diffusion_model_type == "unet"
            for t in self.scheduler.timesteps:
                eps = (self.model.noise_pred_net(a, t, global_cond=cond) if unet
                       else self.model.noise_pred_net(a, t, cond=feats))
                a = self.scheduler.step(eps, t, a).prev_sample
            a = a[0].double().cpu().numpy()

        out = np.repeat(proprio[-1:].astype(np.float64), self.action_horizon, axis=0)
        cols = {"left": slice(0, 10), "right": slice(10, 20)}
        if len(self.arms) == 2:
            parts = {"left": a[:, :10], "right": a[:, 10:]}
        else:
            parts = {self.arms[0]: a}
        for side, part in parts.items():
            c = cols[side]
            out[:, c] = (part + 1.0) / 2.0 * (self.action_max[c] - self.action_min[c]) \
                + self.action_min[c]
        return out
