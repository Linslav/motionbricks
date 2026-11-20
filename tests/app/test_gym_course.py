"""Replay the bundled keyboard recordings through the native SOMA viewer."""
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import warp as wp

REPLAYS = Path(__file__).resolve().parents[1] / "replays"


@unittest.skipUnless(os.getenv("MOTIONBRICKS_MPS_VIEWER_TESTS") == "1", "requires native SOMA gym")
class GymCourseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import newton.examples
        from app.scene_preview import ScenePreview, _DEFAULT_SCENE
        from app.utils.cli import create_gl_parser

        cls.device = wp.ScopedDevice("cpu")
        cls.device.__enter__()
        cls.addClassCleanup(cls.device.__exit__, None, None, None)
        with patch.object(sys, "argv", ["gym-test"]):
            cls.viewer, _ = newton.examples.init(create_gl_parser())
        cls.addClassCleanup(cls.viewer.close)
        cls.preview = ScenePreview(cls.viewer, _DEFAULT_SCENE)
        cls.addClassCleanup(cls.preview.close)

    def replay(self, name, recording=None):
        from app.scene_preview import _DEFAULT_SCENE
        from tools.gym_replay import play
        if recording is None:
            recording = json.loads((REPLAYS / f"{name}.json").read_text())
        path, scene, assets, pool = self.preview._prepare_scene(_DEFAULT_SCENE)
        scene.character_start.transform = wp.transform(wp.vec3(*recording["spawn"]), wp.quat_identity())
        self.preview._start_scene(path, scene, assets, pool)
        rows, _ = play(self.preview, recording)
        self.assertEqual(len(rows), recording["frames"])
        return rows

    def test_repeated_reversals(self):
        rows = self.replay("soma_reversals")
        for index, frame in enumerate(range(54, 174, 12)):
            direction = 1 if index % 2 else -1
            self.assertGreater(direction * rows[frame + 1]["vx"], 0.5)
            self.assertGreater(direction * (rows[frame + 11]["x"] - rows[frame - 1]["x"]), 0.87)
        self.assertFalse(any(row["action"] for row in rows))

    def test_open_floor_starts_stops_and_sprint_turns(self):
        rows = self.replay("open_floor")
        for frame, expected in ((7, 4.5), (21, 2.3), (31, 0.0), (38, -2.3),
                                (49, -4.5), (64, 4.5), (78, -4.5), (91, 0.0)):
            with self.subTest(frame=frame):
                self.assertAlmostEqual(rows[frame]["vx"], expected, delta=0.1)

    def test_wall_blocks_then_space_vaults(self):
        rows = self.replay("vault")
        self.assertLess(max(r["x"] for r in rows[:42]), 3.70)
        self.assertFalse(any(r["action"] for r in rows[:42]))
        self.assertEqual(rows[42]["action"], "MidJumpSP")
        self.assertGreater(rows[-1]["x"], 4.35)
        self.assertGreater(max(r["hips_z"] for r in rows[42:]), rows[35]["hips_z"] + 0.3)

    def test_seat_blocks_then_space_sits_and_movement_stands(self):
        rows = self.replay("seat")
        self.assertLess(max(r["x"] for r in rows[:53]), 3.63)
        self.assertFalse(any(r["action"] for r in rows[:53]))
        self.assertTrue(any(r["seated"] for r in rows[53:180]))
        self.assertFalse(rows[-1]["action"])

    def test_short_long_crouched_and_moving_jumps(self):
        rows = self.replay("airborne")
        peaks = [max(r["z"] for r in rows[a:b]) for a,b in [(0,38),(38,78),(83,123),(132,178)]]
        self.assertGreater(peaks[0], 0.3)
        self.assertGreater(peaks[1], peaks[0] + 0.25)
        self.assertGreater(min(peaks[2:]), 0.65)
        self.assertTrue(rows[-1]["grounded"])
        self.assertFalse(any(r["action"] for r in rows))

    def test_press_and_release_between_frames(self):
        rows = self.replay(None, dict(fps=30, frames=70, spawn=[-8,4,0], events=[[0,[]]], taps=[[3,"space"]]))
        self.assertGreater(max(r["z"] for r in rows), 0.3)
        self.assertTrue(rows[-1]["grounded"])
        self.assertEqual(sum(r["grounded"] and not s["grounded"] for r,s in zip(rows,rows[1:])), 1)

    def test_landing_block_and_headroom(self):
        rows = self.replay(None, dict(fps=30, frames=65, spawn=[-2.7,10,0],
                                 events=[[0, ["shift","space","w"]], [12, []]]))
        self.assertAlmostEqual(rows[-1]["z"], 0.46, places=2)
        self.assertTrue(rows[-1]["grounded"])
        rows = self.replay(None, dict(fps=30, frames=50, spawn=[3,16,0],
                                 events=[[0, ["ctrl","shift","w"]], [12, ["w"]], [32, []]]))
        self.assertEqual(rows[13]["mode"], "crouch_directional_2d")
        self.assertEqual(rows[-1]["mode"], "linear_1d")

    def test_press_space_during_walk_run_and_crouch_at_wall(self):
        for movement in (["w"], ["shift","w"], ["ctrl","w"]):
            with self.subTest(movement=movement):
                rows = self.replay(None, dict(fps=30, frames=180, spawn=[2.5,8,0],
                    events=[[0, movement], [7, sorted(movement + ["space"])], [8, []]]))
                self.assertFalse(any(row["action"] for row in rows[:7]))
                self.assertEqual(rows[7]["action"], "MidJumpSP")
                self.assertGreater(rows[-1]["x"], 4.35)

    def test_low_hurdle(self):
        rows = self.replay("hurdle")
        self.assertGreater(rows[-1]["x"], -2.5)
        self.assertFalse(any(r["action"] for r in rows))
        self.assertTrue(any(not r["grounded"] for r in rows))

    def test_carry_up_and_down_both_ramps(self):
        rows = self.replay("carry_bridge")
        self.assertTrue(all(r["carrying"] and r["grounded"] for r in rows[1:117]))
        self.assertAlmostEqual(max(r["z"] for r in rows[:117]), 1.2, places=2)
        self.assertGreater(rows[116]["x"], 3.0)
        self.assertAlmostEqual(rows[116]["z"], 0.0, places=2)
        self.assertFalse(rows[-1]["carrying"])

    def test_carry_while_running_crouching_and_jumping(self):
        rows = self.replay("carry_free_jump")
        self.assertTrue(all(r["carrying"] for r in rows))
        self.assertEqual(sum(r["grounded"] and not s["grounded"] for r,s in zip(rows,rows[1:])), 3)

    def test_carry_through_vault_and_sitting(self):
        rows = self.replay("carry_vault")
        self.assertTrue(all(r["carrying"] for r in rows))
        self.assertTrue(any(r["action"] == "MidJumpSP" for r in rows))
        self.assertGreater(rows[-1]["x"], 5.0)
        rows = self.replay("carry_seat")
        self.assertTrue(any(r["carrying"] and r["seated"] for r in rows))
        self.assertFalse(rows[-1]["action"])
        self.assertFalse(rows[-1]["carrying"])
