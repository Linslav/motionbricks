"""Record keyboard transitions or replay a trace in the native SOMA gym."""

import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np
import newton.examples
import warp as wp
from pyglet.window import key

from app.scene_preview import ScenePreview, _DEFAULT_SCENE
from app.utils.cli import create_gl_parser

KEYS = {name: getattr(key, name.upper()) for name in ("w", "a", "s", "d", "up", "down", "left", "right", "space", "e", "p", "o", "i")}
KEYS.update(shift=key.LSHIFT, ctrl=key.LCTRL, rshift=key.RSHIFT, rctrl=key.RCTRL, style=key.BRACKETLEFT)


def play(preview, recording, *, record=False):
    """Use a fixed simulation step. Wall time measures the cost of each rendered step."""
    fps = recording["fps"]
    window = preview.viewer.renderer.window
    events = dict(recording["events"])
    observed = []
    taps = {}
    for index, name in recording.get("taps", []):
        taps.setdefault(index, []).append(name)
    observed_taps = []
    rows = []
    previous = set()
    for frame in range(recording["frames"]):
        if not preview.viewer.is_running():
            break
        if not record and frame in events:
            wanted = set(events[frame])
            for name in sorted(wanted ^ previous):
                window.dispatch_event("on_key_press" if name in wanted else "on_key_release", KEYS[name], 0)
        for name in taps.get(frame, []):
            window.dispatch_event("on_key_press", KEYS[name], 0)
            window.dispatch_event("on_key_release", KEYS[name], 0)
        window.dispatch_events()
        held = {name for name, code in KEYS.items() if code in preview._held_keys}
        if not frame or held != previous:
            observed.append([frame, sorted(held)])
        for name in ("space", "e"):
            if name not in held and KEYS[name] in preview._pressed_keys:
                observed_taps.append([frame, name])
        previous = held
        start = time.perf_counter()
        preview.step(1 / fps)
        preview.render()
        elapsed = time.perf_counter() - start
        session = preview._scene_session
        char = session.character
        motor = getattr(char, "motor", None)
        action = preview._active_interaction
        pose = char.current_pose.numpy()
        if not np.isfinite(pose).all():
            raise RuntimeError(f"Non-finite character pose at frame {frame}")
        rows.append(dict(frame=frame, hips_z=float(pose[0, 2]), x=float(char.character_ground_position[0]), y=float(char.character_ground_position[1]),
                         z=float(char.character_ground_position[2]), vx=float(char.actual_velocity[0]),
                         vy=float(char.actual_velocity[1]), mode=session.state_machine.active_locomotion_state.active_mode.name,
                         action=action.definition.name if action else "", seated=session.state_machine.interaction_state.is_looping,
                         grounded=motor.grounded if motor else None, carrying=getattr(char, "carrying", False),
                         frame_ms=1000 * elapsed))
        if record:
            time.sleep(max(0, 1 / fps - elapsed))
    if record and observed_taps:
        recording["taps"] = observed_taps
    if not record and (observed != recording["events"] or observed_taps != recording.get("taps", [])):
        raise RuntimeError("The viewer received different keys during replay.")
    return rows, observed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--replay", type=Path)
    source.add_argument("--record", type=Path, help="Record a live keyboard session to a new JSON file")
    parser.add_argument("--seconds", type=float, default=20)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--acceleration", type=float, help="Override horizontal acceleration for a tuning replay (m/s²)")
    parser.add_argument("--reference", action="store_true")
    args = parser.parse_args()
    recording = (json.loads(args.replay.read_text()) if args.replay else
                 dict(fps=30, frames=round(30 * args.seconds), spawn=[-8, 4, 0], events=[]))
    if args.record and args.record.exists():
        parser.error("Choose a new recording filename.")
    # Newton reads its own command line. Isolate the replay arguments.
    import sys
    sys.argv = sys.argv[:1]
    with wp.ScopedDevice("cpu"):
        viewer, _ = newton.examples.init(create_gl_parser())
        scene = ScenePreview(viewer, _DEFAULT_SCENE, reference=args.reference)
        try:
            path, definition, assets, pool = scene._prepare_scene(_DEFAULT_SCENE)
            definition.character_start.transform = wp.transform(wp.vec3(*recording["spawn"]), wp.quat_identity())
            scene._start_scene(path, definition, assets, pool)
            if args.acceleration is not None:
                if args.reference or args.acceleration <= 0:
                    parser.error("--acceleration needs a positive value and game controls")
                scene._scene_session.character.motor.acceleration = args.acceleration
            rows, observed = play(scene, recording, record=bool(args.record))
            if args.record:
                recording.update(events=observed, frames=len(rows))
                args.record.write_text(json.dumps(recording, indent=2) + "\n")
            if args.csv and rows:
                with args.csv.open("w", newline="") as out:
                    writer = csv.DictWriter(out, fieldnames=rows[0])
                    writer.writeheader()
                    writer.writerows(rows)
            print(f"{len(rows)} frames, {len(observed)} keyboard changes")
        finally:
            scene.close()
            viewer.close()


if __name__ == "__main__":
    main()
