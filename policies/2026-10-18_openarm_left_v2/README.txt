OpenArm cup policy, LEFT arm, v2 epoch 6 (~5.6k training steps, raw weights), commit 2fc0cfa4

FOR DEPLOYMENT PLUMBING TESTS ONLY. In the closed-loop sim eval this checkpoint did not grasp the cup yet (0/2 episodes).
Keep the e-stop at hand and expect it to miss the cup.

Files
  openarm_cup_policy_6_model.pt    the self-contained policy file (weights, config, action stats, calibration, interface)
  openarm_cup_policy_6_model.json  the same metadata, readable
  openarm_policy.py, tinydp_vision.py  the loader (OpenArmCupPolicy); tinydp_vision.py builds the spatial-softmax ResNet

Camera input: raw 1920x1080 BGR -> undistort (bundled calibration) -> crop (300, 0, 1280, 1080) -> policy crop
  [320, 360, 960, 720] -> resize to 320x320. OpenArmCupPolicy.preprocess(frame) does all of it.
Rates: 15 Hz observations and actions, 4 observations in, 16-step chunks out (left TCP xyz + quat wxyz, gripper).

Robot start (OPENARM_HOME_LEFT_ACTIVE), left joints 1-7:
  [0.02, -0.428, 0.107, 2.196, -0.467, 0.08, 0.547]; right arm hanging (all zero).
Cup start: upright, centre within 0.12 m of [0.541, 0.076] (robot frame),
  i.e. [0.075, 0.082] from the table centre; handle towards the robot (173.9 deg, +-45.0).
