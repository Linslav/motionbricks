"""Kinematic player movement against the boxes visible in the SOMA gym."""

import math

import numpy as np
from scipy.spatial.transform import Rotation


class BoxSurface:
    """An oriented top face and its footprint in the horizontal plane."""

    def __init__(self, cube):
        rotation = Rotation.from_quat(tuple(cube.transform.q)).as_matrix()
        self.normal = rotation[:, 2]
        self.top = np.array(cube.transform.p) + self.normal * cube.height / 2
        self.xy_to_face = np.linalg.inv(rotation[:2, :2])
        self.half = np.array((cube.width, cube.depth)) / 2
        self.thickness = cube.height / self.normal[2]
        self.extent = np.abs(rotation[:2, :2]) @ self.half

    def span(self, x, y, margin=0.0):
        dx, dy = x - self.top[0], y - self.top[1]
        if abs(dx) > self.extent[0] + margin or abs(dy) > self.extent[1] + margin:
            return None
        u, v = self.xy_to_face @ (dx, dy)
        if abs(u) > self.half[0] + margin or abs(v) > self.half[1] + margin:
            return None
        z = self.top[2] - (self.normal[0] * dx + self.normal[1] * dy) / self.normal[2]
        return z - self.thickness, z


class GymMotor:
    standing_height = 1.77
    crouching_height = 1.15
    radius = 0.19
    step_height = 0.10
    acceleration = 110.0  # m/s², independent of how fast the pose can turn
    gravity = 15.0
    launch_speed = 5.7

    def __init__(self, position, cubes):
        self.position = np.array(position, dtype=float)
        self.velocity = np.zeros(2)
        self.vertical_speed = 0.0
        self.grounded = True
        self.buffer = 0.0
        self.jump_cut = False
        self.surfaces = [BoxSurface(cube) for cube in cubes]

    def floor(self, x, y, limit):
        height = 0.0
        for surface in self.surfaces:
            span = surface.span(x, y)
            if span and span[1] <= limit:
                height = max(height, span[1])
        return height

    def clear(self, x, y, feet, height):
        for surface in self.surfaces:
            span = surface.span(x, y, self.radius)
            if span and span[1] > feet + self.step_height and span[0] < feet + height - 0.001:
                return False
        return True

    def separate_after_interaction(self, height):
        """Exit an authored stand/vault that ends inside a gameplay collider."""
        x, y, z = self.position
        if self.clear(x, y, z, height):
            return
        candidates = []
        for box in self.surfaces:
            span = box.span(x, y, self.radius)
            if not span or span[1] <= z + self.step_height or span[0] >= z + height:
                continue
            local = box.xy_to_face @ (self.position[:2] - box.top[:2])
            for axis in (0, 1):
                for sign in (-1, 1):
                    edge = local.copy()
                    edge[axis] = sign * (box.half[axis] + self.radius + 0.002)
                    xy = box.top[:2] + np.linalg.solve(box.xy_to_face, edge)
                    if self.clear(*xy, z, height):
                        candidates.append(xy)
        if candidates:
            self.position[:2] = min(candidates, key=lambda xy: np.linalg.norm(xy - self.position[:2]))

    def jump(self, held):
        # A tap between two rendered frames still produces a short hop.
        self.buffer = 0.13
        self.jump_cut = not held

    def advance(self, desired, dt, crouching, jump_held):
        height = self.crouching_height if crouching else self.standing_height
        steps = max(1, math.ceil(dt * 120))
        h = dt / steps
        for _ in range(steps):
            if self.buffer > 0 and self.grounded:
                self.vertical_speed = self.launch_speed
                self.grounded = False
                self.buffer = 0.0
            self.buffer = max(0.0, self.buffer - h)
            if not jump_held or self.jump_cut:
                self.vertical_speed = min(self.vertical_speed, 3.6)
            delta = desired - self.velocity
            length = np.linalg.norm(delta)
            self.velocity += delta * min(1.0, self.acceleration * h / max(length, 1e-8))
            x, y, z = self.position
            for axis in (0, 1):
                nx = x + self.velocity[0] * h if axis == 0 else x
                ny = y + self.velocity[1] * h if axis == 1 else y
                if self.clear(nx, ny, z, height):
                    x, y = nx, ny
                else:
                    self.velocity[axis] = 0.0
            floor = self.floor(x, y, z + self.step_height)
            if self.grounded and z - floor <= self.step_height:
                z = floor
            else:
                self.grounded = False
                self.vertical_speed -= self.gravity * h
                next_z = z + self.vertical_speed * h
                if self.vertical_speed > 0:
                    for surface in self.surfaces:
                        span = surface.span(x, y, self.radius)
                        if span and z + height <= span[0] + 0.001 and next_z + height >= span[0]:
                            next_z = span[0] - height
                            self.vertical_speed = 0.0
                z = max(floor, next_z)
                if z <= floor and self.vertical_speed <= 0:
                    self.grounded = True
                    self.vertical_speed = 0.0
            self.position[:] = x, y, z
