# Changelog

## Unreleased

- **Add responsive traversal and manual actions to the SOMA gym** — Sprint reversals respond within two replay frames. Space jumps, vaults or sits. The course adds solid obstacles, ramps, a landing block, a hand-carried baton, and recordable keyboard trials.

- **Tighten locomotion and add held sprint and crouch controls** — Movement targets respond on the first step, turns settle sooner, and stops take less space. Hold Shift to run and Control to crouch.

- **Stand up from the seat with movement input** — Press a movement key or move the controller stick to get up, then keep moving to walk away.

## [0.2.0] - 2026-10-23

- MotionBricks moves to its own repository. It started as a Unitree G1 preview in
  [GR00T Whole-Body Control](https://github.com/NVlabs/GR00T-WholeBodyControl/tree/main/motionbricks).
- A new pretrained model, retrained on the SOMA human skeleton.
- The paper's Smart Primitive setup, rebuilt in plain Python with Newton, Warp, and TensorRT.
- New [Inbetweener](README.md#inbetweener) and [Scene Preview](README.md#scene-preview) demos for generating continuous motion and interacting with scenes.
