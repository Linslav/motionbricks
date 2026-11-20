"""Immediate player travel with sampled SOMA gaits and generated object interactions."""

import math

import numpy as np
import warp as wp

from motionbricks.animation.ik import solve_two_bone_ik
from motionbricks.utils.pose_utils import blend_transform
from .character import SOMACharacter
from .gym_motor import GymMotor
from ..common.io.smart_primitive_document import load_smart_primitive_document
from ..common.motion import CLIP_FACING_YAW, CROUCH_LOCOMOTION_MODE, LINEAR_LOCOMOTION_MODE, PlaybackSource
from ..common.rig import ASSETS_ROOT
from ..common.state_machine import StateMachine
from ..common.states import LocomotionState


@wp.kernel
def _place_gait(pose: wp.array(dtype=wp.transform), previous: wp.array(dtype=wp.transform),
                feet: wp.vec3, yaw: float, blend: float, airborne: bool):
    joint = wp.tid()
    value = pose[joint]
    if joint == 0:
        height = value.p[2]
        if airborne:
            # The motor supplies the jump arc. The clip supplies the tucked limbs.
            height = 0.94
        rotation = wp.quat_from_axis_angle(wp.vec3(0., 0., 1.), yaw)
        value = wp.transform(feet + wp.vec3(0., 0., height), rotation * value.q)
        mixed = blend_transform(previous[joint], value, blend)
        value = wp.transform(wp.vec3(feet[0], feet[1], mixed.p[2]), mixed.q)
    else:
        value = blend_transform(previous[joint], value, blend)
    pose[joint] = value


@wp.kernel
def _fit_feet(pose: wp.array(dtype=wp.transform), world: wp.array(dtype=wp.transform),
              parents: wp.array(dtype=wp.int32), feet: wp.array(dtype=wp.int32),
              offsets: wp.vec2, normal: wp.vec3):
    i = wp.tid()
    tip = feet[i]
    mid = parents[tip]
    leg = parents[mid]
    parent = world[parents[leg]]
    foot = world[tip]
    up = wp.vec3(0., 0., 1.)
    target = wp.transform(foot.p + up * offsets[i], wp.quat_between_vectors(up, normal) * foot.q)
    a, b, c = solve_two_bone_ik(1., parent, world[leg], world[mid], foot, target, False, world[mid].p)
    pose[leg] = wp.mul(wp.transform_inverse(parent), a)
    pose[mid] = wp.mul(wp.transform_inverse(a), b)
    pose[tip] = wp.mul(wp.transform_inverse(b), c)


class _GaitState(LocomotionState):
    def enter(self, character, state_machine):
        pass  # The motor samples the selected gait every frame, including entry.

    def update(self, character, state_machine, delta_time):
        if character.movement.input_speed <= 0.0:
            state_machine.transition_to_idle(character)


class _GymStates(StateMachine):
    locomotion_type = _GaitState


class GymCharacter(SOMACharacter):
    state_machine_type = _GymStates

    def __init__(self, engine, *, scene, **kwargs):
        super().__init__(engine, **kwargs)
        self.prop = next((cube for cube in scene.solid_cubes if cube.name == "Relay baton"), None)
        self.motor = GymMotor(self.character_ground_position, [c for c in scene.solid_cubes if c is not self.prop])
        self.carrying = False
        self.jump_held = False
        self._world = wp.empty_like(self.current_pose)
        self._ramps = [box for box in self.motor.surfaces if box.normal[2] < 0.999]
        self._foot_indices = [self.skeleton.joint_index(name) for name in ("LeftFoot", "RightFoot")]
        self._feet = wp.array(self._foot_indices, dtype=wp.int32, device=engine.device)
        self._right_hand = self.skeleton.joint_index("RightHand")
        self._previous = wp.clone(self.current_pose)
        self._phase = 0.0
        self._yaw = math.atan2(*self.movement.current_facing[1::-1])
        self._was_interacting = False
        self._previous_animation = None
        self._blend_time = 0.0
        self._gaits = {}
        modes = self.state_machine.standing_locomotion_state.mode_runtimes.copy()
        modes.update(self.state_machine.crouching_locomotion_state.mode_runtimes)
        for path in {m.definition.smart_primitive_path for m in modes.values()}:
            doc, _, _ = load_smart_primitive_document(self.clip_registry, self.asset_pool.reference_pose,
                                                       self.model_to_runtime, path)
            for node in doc.root.children:
                target = node.target_pose
                if target and target.clip and node.name != "Idle":
                    self._gaits[path, node.name] = self._stride(target.clip.animation, target.sample_time)
        jump = ASSETS_ROOT / "motions/soma/jump/jump_over_obstacle_0_5m_R_001__A415.bvh"
        self._jump = self.clip_registry.load(jump).animation

    def _stride(self, animation, start):
        """Find a matching leg pose one stride later, near the authored gait sample."""
        joints = [self.skeleton.joint_index(name) for name in ("LeftLeg", "RightLeg", "LeftShin", "RightShin")]
        rate = animation.sample_rate
        first = max(0, min(round(start * rate), animation.num_frames - 1 - round(1.45 * rate)))
        rotations = animation.poses.numpy()[:, joints, 3:]
        low = first + round(0.48 * rate)
        high = min(animation.num_frames, first + round(1.45 * rate))
        match = np.abs((rotations[low:high] * rotations[first]).sum(axis=2)).mean(axis=1)
        end = low + int(np.argmax(match))
        return animation, first / rate, (end - first) / rate

    def generate_clip_transition(self, clip):
        self._current_clip = clip
        self._playback_source = PlaybackSource.CLIP
        self._clip_loop = True
        self._clip_playback_time = 0.0

    def _gait(self, mode):
        if not self.motor.grounded:
            # Play the authored airborne poses through ascent and descent.
            flight = min(1.0, max(0.0, (self.motor.launch_speed - self.motor.vertical_speed) / (2 * self.motor.launch_speed)))
            return self._jump, 1.90 + 0.55 * flight
        if not self.state_machine.is_locomoting:
            clip = self.current_clip
            return clip.animation, self._clip_playback_time % clip.duration
        if mode.name == LINEAR_LOCOMOTION_MODE:
            label = "Run" if self.movement.input_speed > 3.0 else "Walk"
        elif len(mode.target_values) == 2:
            label = mode.target_values[-1][0]
        else:
            dx, dy = self.movement.input_velocity_direction[:2]
            fx, fy = self.movement.desired_facing[:2]
            forward, left = dx * fx + dy * fy, dy * fx - dx * fy
            if abs(forward) >= abs(left):
                label = "Forward" if forward >= 0 else "Backward"
            else:
                label = "Left" if left >= 0 else "Right"
        animation, start, duration = self._gaits[mode.smart_primitive_path, label]
        return animation, start + self._phase % duration

    def _sample_playback(self, dt):
        if self.state_machine.is_interacting:
            self._was_interacting = True
            self._previous_animation = None
            super()._sample_playback(dt)
            self.motor.position[:] = self.character_ground_position
            self.motor.velocity[:] = 0
            return
        mode = self.state_machine.active_locomotion_state.active_mode
        crouching = mode.name == CROUCH_LOCOMOTION_MODE
        if self._was_interacting:
            self.motor.separate_after_interaction(self.motor.crouching_height if crouching else self.motor.standing_height)
            self._was_interacting = False
        self.motor.advance(self.movement.input_velocity[:2], dt, crouching, self.jump_held)
        self.character_ground_position[:] = self.motor.position
        self._clip_playback_time += dt
        self._phase += dt * (self.movement.input_speed / mode.default_speed if crouching else 1.0)
        animation, sample = self._gait(mode)
        if animation is not self._previous_animation:
            wp.copy(self._previous, self.current_pose)
            self._previous_animation = animation
            self._blend_time = 0.
        self._blend_time = min(0.10, self._blend_time + dt)
        direction = self.movement.desired_facing if self.movement.input_speed else self.movement.last_input_facing
        target = math.atan2(direction[1], direction[0])
        delta = math.remainder(target - self._yaw, 2 * math.pi)
        self._yaw += np.clip(delta, -dt * 22, dt * 22)
        animation.sample(float(sample), self.current_pose, root_offset=self.model_to_runtime, stream=self.engine.stream)
        wp.launch(_place_gait, dim=self.skeleton.num_joints, device=self.engine.device, stream=self.engine.stream,
                  inputs=[self.current_pose, self._previous, wp.vec3(*self.motor.position),
                          float(self._yaw - CLIP_FACING_YAW), self._blend_time / 0.10, not self.motor.grounded])
        if self.motor.grounded:
            self._fit_slope()
        self.current_foot_contacts.zero_()

    def _fit_slope(self):
        x, y, z = self.motor.position
        for ramp in self._ramps:
            span = ramp.span(x, y)
            if span is None or abs(z - span[1]) > 0.02:
                continue
            self.skeleton.compute_world_transforms(self.current_pose, self._world, stream=self.engine.stream)
            feet = self._world.numpy()[self._foot_indices, :2]
            offsets = [self.motor.floor(*xy, z + 0.28) - z for xy in feet]
            wp.launch(_fit_feet, dim=2, device=self.engine.device, stream=self.engine.stream,
                      inputs=[self.current_pose, self._world, self.skeleton.parent_indices, self._feet,
                              wp.vec2(*np.clip(offsets, -0.18, 0.18)), wp.vec3(*ramp.normal)])
            break

    def toggle_prop(self):
        if self.prop is None:
            return
        if self.carrying:
            x, y = self.character_ground_position[:2] + self.movement.current_facing[:2] * 0.50
            z = self.motor.floor(x, y, self.motor.position[2] + 0.12)
            self.prop.transform = wp.transform(wp.vec3(x, y, z + self.prop.height / 2), wp.quat_identity())
            self.carrying = False
        elif np.linalg.norm(np.asarray(self.prop.transform.p) - self.motor.position) <= 1.65:
            self.carrying = True

    def step(self, dt, command):
        super().step(dt, command)
        if self.carrying:
            self.skeleton.compute_world_transforms(self.current_pose, self._world, stream=self.engine.stream)
            wrist = wp.transform(*self._world[self._right_hand:self._right_hand + 1].numpy()[0])
            self.prop.transform = wp.mul(wrist, wp.transform(wp.vec3(-0.079, 0, -0.014), wp.quat_identity()))
