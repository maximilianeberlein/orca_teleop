"""Collect human hand motion data for GeoRT training.

Records hand landmarks from any supported input source (mediapipe, manus)
and saves them as a numpy array in GeoRT's canonical wrist-centered frame.

Usage:
    python scripts/geort/collect_human_data.py --source mediapipe --name my_data --model-path path/to/model

Output: [N_frames, N_landmarks, 3] numpy array saved to third_party/GeoRT/data/<name>.npy
"""
import os
import sys
import time
import argparse
import threading
import numpy as np

# Add project root to path
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.join(_SCRIPT_DIR, "..", "..")
sys.path.insert(0, _PROJECT_ROOT)

_GEORT_ROOT = os.path.join(_PROJECT_ROOT, "third_party", "GeoRT")
sys.path.insert(0, _GEORT_ROOT)

from orca_teleop.orca_retargeter.utils import retargeter_utils

# Fingertip indices per source (used for viewer highlights)
FINGERTIP_INDICES = {
    "mediapipe": [4, 8, 12, 16, 20],
    "manus": [24, 4, 9, 19, 14],
}

# Colors per finger: thumb, index, middle, ring, pinky
FINGER_COLORS = np.array([
    [255, 80, 80],     # thumb — red
    [80, 255, 80],     # index — green
    [80, 80, 255],     # middle — blue
    [255, 255, 80],    # ring — yellow
    [255, 80, 255],    # pinky — magenta
], dtype=np.uint8)

# canonical→palm rotation (matches train_ik_standalone.py line 126-127)
CANONICAL_TO_PALM = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=np.float32)


def _load_robot_pointcloud(hand_config_name):
    """Load robot FK pointcloud from GeoRT data for overlay."""
    npz_path = os.path.join(_GEORT_ROOT, "data", f"{hand_config_name}.npz")
    if not os.path.exists(npz_path):
        return None, None
    d = np.load(npz_path, allow_pickle=True)
    kp = d["keypoint"].item()
    finger_order = [
        "right_thumb_fingertip", "right_index_fingertip",
        "right_middle_fingertip", "right_ring_fingertip",
        "right_pinky_fingertip"
    ]
    points = []
    colors = []
    for i, link in enumerate(finger_order):
        if link in kp:
            pts = kp[link]
            # Subsample for display
            n = min(2000, len(pts))
            idx = np.random.choice(len(pts), n, replace=False)
            points.append(pts[idx])
            colors.append(np.tile(FINGER_COLORS[i] // 3, (n, 1)))  # dim colors for backdrop
    if not points:
        return None, None
    return np.concatenate(points, axis=0), np.concatenate(colors, axis=0).astype(np.uint8)


def _start_viewer(source, hand_config_name):
    """Start a viser 3D viewer showing robot pointcloud and live hand landmarks."""
    import viser

    server = viser.ViserServer(port=8890)
    server.scene.set_up_direction("+y")

    # Load and display robot reachable positions (in palm frame) as dim pointcloud
    robot_pts, robot_colors = _load_robot_pointcloud(hand_config_name)
    if robot_pts is not None:
        # Robot data is in palm-local frame already
        server.scene.add_point_cloud(
            "/robot_reachable",
            points=robot_pts.astype(np.float32),
            colors=robot_colors,
            point_size=0.002,
            point_shape="circle",
        )

    # Add axis frame at origin for reference
    server.scene.add_frame("/origin", axes_length=0.05, axes_radius=0.002)

    # Placeholder for live hand points (updated from callback)
    tip_indices = FINGERTIP_INDICES[source]
    live_handle = [None]  # mutable container for the point cloud handle
    tip_handles = [None] * 5
    line_handle = [None]

    def update_landmarks(canonical_points):
        """Update the viewer with new canonical-frame landmarks."""
        # Transform canonical→palm frame (same as IK training)
        palm_pts = canonical_points @ CANONICAL_TO_PALM

        # All landmarks as white points
        n_pts = len(palm_pts)
        all_colors = np.full((n_pts, 3), 200, dtype=np.uint8)

        # Color the fingertips
        for i, tip_idx in enumerate(tip_indices):
            if tip_idx < n_pts:
                all_colors[tip_idx] = FINGER_COLORS[i]

        if live_handle[0] is not None:
            live_handle[0].remove()
        live_handle[0] = server.scene.add_point_cloud(
            "/live_hand",
            points=palm_pts.astype(np.float32),
            colors=all_colors,
            point_size=0.008,
            point_shape="circle",
        )

        # Draw fingertip spheres (larger, colored)
        for i, tip_idx in enumerate(tip_indices):
            if tip_idx < n_pts:
                if tip_handles[i] is not None:
                    tip_handles[i].remove()
                tip_handles[i] = server.scene.add_icosphere(
                    f"/tips/{i}",
                    radius=0.005,
                    color=tuple(FINGER_COLORS[i].tolist()),
                    position=palm_pts[tip_idx].astype(np.float32),
                )

        # Draw hand skeleton lines (only for MediaPipe 21-point layout)
        if source == "mediapipe" and n_pts >= 21:
            connections = [
                (0, 1), (1, 2), (2, 3), (3, 4),
                (0, 5), (5, 6), (6, 7), (7, 8),
                (0, 9), (9, 10), (10, 11), (11, 12),
                (0, 13), (13, 14), (14, 15), (15, 16),
                (0, 17), (17, 18), (18, 19), (19, 20),
                (5, 9), (9, 13), (13, 17),
            ]
            starts = np.array([palm_pts[a] for a, b in connections], dtype=np.float32)
            ends = np.array([palm_pts[b] for a, b in connections], dtype=np.float32)
            if line_handle[0] is not None:
                line_handle[0].remove()
            line_handle[0] = server.scene.add_line_segments(
                "/hand_skeleton",
                points=np.stack([starts, ends], axis=1).reshape(-1, 3),
                colors=np.full((len(connections) * 2, 3), 180, dtype=np.uint8),
                line_width=2.0,
            )

    return server, update_landmarks


def main():
    parser = argparse.ArgumentParser(description="Collect human hand data for GeoRT training")
    parser.add_argument("--source", type=str, required=True, choices=["mediapipe", "manus"],
                        help="Input source type")
    parser.add_argument("--name", type=str, default="human_data",
                        help="Dataset name (saved to third_party/GeoRT/data/<name>.npy)")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to ORCA hand model (for ingress initialization)")
    parser.add_argument("--duration", type=float, default=300.0,
                        help="Max recording duration in seconds (default: 5 minutes)")
    parser.add_argument("--hand", type=str, default="orca_right_new_index",
                        help="GeoRT hand config name (for robot pointcloud overlay)")
    parser.add_argument("--no-viewer", action="store_true",
                        help="Disable 3D viewer")
    # Manus-specific args
    parser.add_argument("--glove-id", type=str, default=None,
                        help="Manus glove hex ID (required for --source manus)")
    parser.add_argument("--zmq-addr", type=str, default="tcp://localhost:8000",
                        help="ZMQ address for Manus SDK stream")
    args = parser.parse_args()

    # Start viewer
    viewer_server = None
    update_viewer = None
    if not args.no_viewer:
        try:
            viewer_server, update_viewer = _start_viewer(args.source, args.hand)
            print("3D viewer running at http://localhost:8890")
            print("  Open browser and scroll to zoom into the origin area")
            print("  Dim pointcloud = robot reachable positions")
            print("  Bright points = your live hand landmarks")
            print("  Colored spheres = fingertips the IK model uses")
            # Check if robot pointcloud loaded
            robot_pts, _ = _load_robot_pointcloud(args.hand)
            if robot_pts is None:
                print(f"  WARNING: No robot pointcloud found for '{args.hand}' "
                      f"(missing third_party/GeoRT/data/{args.hand}.npz)")
        except Exception as e:
            print(f"Warning: could not start viewer: {e}")

    frames = []
    recording = False
    last_viewer_update = [0.0]
    callback_count = [0]

    def on_landmarks(data):
        nonlocal recording
        if args.source == "mediapipe":
            joints, _ = retargeter_utils.preprocess_mediapipe_data({"hand_landmarks": data})
        elif args.source == "manus":
            joints, _ = retargeter_utils.preprocess_manus_data(data)
        canonical = retargeter_utils.to_geort_canonical_frame(joints, args.source)

        callback_count[0] += 1
        if callback_count[0] == 1:
            print(f"  First callback received! canonical shape={canonical.shape}, "
                  f"range=[{canonical.min():.4f}, {canonical.max():.4f}]")

        # Update viewer (throttled to ~30fps)
        if update_viewer is not None:
            now = time.time()
            if now - last_viewer_update[0] > 0.033:
                last_viewer_update[0] = now
                try:
                    update_viewer(canonical)
                except Exception as e:
                    if callback_count[0] <= 3:
                        print(f"  Viewer error: {e}")

        if recording:
            frames.append(canonical)

    # Initialize ingress
    if args.source == "mediapipe":
        from orca_teleop import MediaPipeIngress
        ingress = MediaPipeIngress(args.model_path, callback=on_landmarks)
    elif args.source == "manus":
        if not args.glove_id:
            parser.error("--glove-id is required for manus source")
        from orca_teleop import ManusIngress
        ingress = ManusIngress(args.model_path, args.glove_id,
                               callback=on_landmarks, zmq_addr=args.zmq_addr)

    ingress.start()
    print(f"\nSource: {args.source}")
    print("The viewer shows your hand BEFORE recording starts — use it to check positioning.")
    print("Press Enter to START recording, then Enter again to STOP.")
    print("Move your hand through full ROM: stretch fingers, make fists, pinch, etc.")

    try:
        input("\nPress Enter to start recording...")
        recording = True
        print("RECORDING... (press Enter to stop)")
        start_time = time.time()

        # Wait for stop signal or timeout
        import select
        while time.time() - start_time < args.duration:
            if select.select([sys.stdin], [], [], 0.5)[0]:
                sys.stdin.readline()
                break
            if len(frames) % 100 == 0 and len(frames) > 0:
                elapsed = time.time() - start_time
                print(f"  Collected {len(frames)} frames ({elapsed:.1f}s)")

    except KeyboardInterrupt:
        pass
    finally:
        recording = False
        ingress.cleanup()
        if viewer_server is not None:
            viewer_server.stop()

    if len(frames) == 0:
        print("No frames collected. Exiting.")
        return

    # Save
    data = np.array(frames)
    save_dir = os.path.join(_GEORT_ROOT, "data")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{args.name}.npy")
    np.save(save_path, data)
    print(f"Saved {len(frames)} frames with shape {data.shape} to {save_path}")


if __name__ == "__main__":
    main()
