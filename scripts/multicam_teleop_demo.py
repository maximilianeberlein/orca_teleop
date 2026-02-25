#!/usr/bin/env python3
"""Multi-camera MediaPipe triangulation teleop demo.

Runs N cameras simultaneously, triangulates 2D MediaPipe detections into 3D,
and retargets to the ORCA hand.

Prerequisites:
  1. Calibrate intrinsics: python scripts/calibrate_cameras.py calibrate --camera-indices 0 2 4 --square-size 0.035
  2. Place ArUco marker on table (visible to all cameras)

Usage:
  python scripts/multicam_teleop_demo.py path/to/model path/to/urdf --camera-indices 0 2 4
"""

import sys
import time
import multiprocessing
import cv2
import argparse
from orca_teleop.orca_ingress.multicam import MultiCamIngress


def robot_control_process_worker(q, stop, ready, model_path, urdf_path=None):
    import time
    try:
        from orca_core import OrcaHand
        from orca_teleop.orca_retargeter.utils.retargeter_utils import urdf_angles_to_physical
        ref_offsets = None
        if urdf_path is not None:
            from orca_teleop.orca_retargeter.utils.urdf_renamer import load_ref_offsets
            ref_offsets = load_ref_offsets(urdf_path)
        hand = OrcaHand(model_path)
        success, message = hand.connect()
        if not success:
            print(f"Robot process: Failed to connect: {message}")
            return
        hand.init_joints()
        ready.set()
        # Slow ramp for first few seconds to protect tendons
        startup_ramp_duration = 2.0  # seconds
        startup_time = time.monotonic()
        last_physical = None
        while not stop.is_set():
            try:
                angles = q.get(timeout=0.1)
                while not q.empty():
                    try:
                        angles = q.get_nowait()
                    except Exception:
                        break
                if angles:
                    physical = urdf_angles_to_physical(angles, ref_offsets)
                    # During startup ramp, interpolate toward target to limit speed
                    elapsed = time.monotonic() - startup_time
                    if elapsed < startup_ramp_duration and last_physical is not None:
                        alpha = min(elapsed / startup_ramp_duration, 1.0)
                        blended = {}
                        for k in physical:
                            blended[k] = last_physical.get(k, physical[k]) * (1 - alpha) + physical[k] * alpha
                        physical = blended
                    hand.set_joint_pos(physical)
                    last_physical = physical
            except Exception:
                continue
    except Exception as e:
        print(f"Robot process error: {e}")
    finally:
        try:
            hand.disable_torque()
            hand.disconnect()
        except Exception:
            pass


def retarget_process_worker(landmark_queue, angles_queue, stop_event, viewer_stopped_event,
                            model_path, urdf_path, retargeter_type, geort_checkpoint, geort_config,
                            enable_viewer, manual_calib, do_debug_timing):
    """Retarget process: owns the retargeter and viewer, runs in its own process to avoid GIL."""
    import time
    import queue

    # Create retargeter inside this process (avoids pickling torch objects)
    if retargeter_type == 'neural-geort':
        from orca_teleop import NeuralGeoRTRetargeter
        retargeter = NeuralGeoRTRetargeter(model_path, urdf_path,
                                           geort_checkpoint=geort_checkpoint,
                                           geort_config=geort_config, source="multicam")
    elif retargeter_type == 'absolute':
        from orca_teleop import AbsoluteRetargeter
        retargeter = AbsoluteRetargeter(model_path, urdf_path, source="multicam")
    elif retargeter_type == 'geort':
        from orca_teleop import GeoRTRetargeter
        retargeter = GeoRTRetargeter(model_path, urdf_path, source="multicam")
    elif retargeter_type == 'pyroki':
        from orca_teleop import PyRoKIRetargeter
        retargeter = PyRoKIRetargeter(model_path, urdf_path, source="multicam")
    else:
        from orca_teleop import Retargeter
        retargeter = Retargeter(model_path, urdf_path, source="multicam")

    viewer = None
    if enable_viewer:
        try:
            from orca_teleop.viewer import URDFViewer
            viewer = URDFViewer(urdf_path, open_browser=False)
            print(f"URDF viewer started at http://localhost:8080")
        except ImportError:
            print("viser not installed — skipping 3D viewer")
        except Exception as e:
            print(f"Failed to start viewer: {e}")

    if manual_calib and viewer:
        viewer.add_calibration_controls(retargeter)

    retargeter.enable_viz = (viewer is not None)

    timing_accum = {"retarget": 0.0, "viewer": 0.0, "total": 0.0}
    timing_count = 0

    while not stop_event.is_set():
        if viewer and viewer.stopped:
            viewer_stopped_event.set()
            break

        try:
            landmarks = landmark_queue.get(timeout=0.05)
        except (queue.Empty, EOFError):
            continue
        # Drain to latest frame
        while True:
            try:
                landmarks = landmark_queue.get_nowait()
            except (queue.Empty, EOFError):
                break

        t_total = time.perf_counter()

        t0 = time.perf_counter()
        angles = retargeter.retarget({"hand_landmarks": landmarks})
        t_retarget = time.perf_counter() - t0

        if angles_queue is not None:
            try:
                angles_queue.put_nowait(angles)
            except Exception:
                pass

        t0 = time.perf_counter()
        if viewer:
            viewer.update(angles)
            if retargeter.mano_points is not None:
                viewer.update_mano_points(retargeter.mano_points)
            if retargeter.fk_points is not None:
                viewer.update_fk_points(retargeter.fk_points)
        t_viewer = time.perf_counter() - t0

        t_total_elapsed = time.perf_counter() - t_total

        if do_debug_timing:
            timing_accum["retarget"] += t_retarget
            timing_accum["viewer"] += t_viewer
            timing_accum["total"] += t_total_elapsed
            timing_count += 1
            if timing_count >= 30:
                n = timing_count
                print(
                    f"[retarget] retarget={timing_accum['retarget']/n*1000:.1f}ms "
                    f"viewer={timing_accum['viewer']/n*1000:.1f}ms "
                    f"total={timing_accum['total']/n*1000:.1f}ms ({n} frames avg)"
                )
                timing_accum = {k: 0.0 for k in timing_accum}
                timing_count = 0

    if viewer:
        viewer.close()


def main():
    parser = argparse.ArgumentParser(description='Multi-camera triangulation teleop demo')
    parser.add_argument('model_path')
    parser.add_argument('urdf_path')
    parser.add_argument('--camera-indices', type=int, nargs='+', default=None,
                        help='Camera /dev/video indices (default: auto-detect)')
    parser.add_argument('--calibration-dir', default='calibration/',
                        help='Directory with cam_N_intrinsics.json files')
    parser.add_argument('--marker-size', type=float, default=0.022,
                        help='ArUco marker side length in meters')
    parser.add_argument('--marker-id', type=int, default=20)
    parser.add_argument('--no-display', action='store_true')
    parser.add_argument('--no-viewer', action='store_true')
    parser.add_argument('--no-robot', action='store_true')
    parser.add_argument('--manual-calib', action='store_true',
                        help='Enable manual calibration sliders in viewer')
    parser.add_argument('--retargeter', choices=['default', 'absolute', 'geort', 'neural-geort', 'pyroki'],
                        default='default')
    parser.add_argument('--geort-checkpoint', type=str, default=None,
                        help='Path to GeoRT IK model checkpoint (.pth)')
    parser.add_argument('--geort-config', type=str, default=None,
                        help='Path to GeoRT config JSON (with joint limits)')
    parser.add_argument('--no-reproj-filter', action='store_true',
                        help='Disable reprojection error refinement')
    parser.add_argument('--jitter-weight', action='store_true',
                        help='Enable temporal jitter weighting (off by default)')
    parser.add_argument('--no-temporal-filter', action='store_true',
                        help='Disable One Euro temporal smoothing')
    parser.add_argument('--debug-timing', action='store_true',
                        help='Print per-stage timing breakdown every 30 frames')
    parser.add_argument('--no-orientation-weight', action='store_true',
                        help='Disable orientation-based per-camera weighting')
    args = parser.parse_args()

    if args.retargeter == 'neural-geort' and (not args.geort_checkpoint or not args.geort_config):
        print("Error: --geort-checkpoint and --geort-config required for neural-geort retargeter")
        return 1

    if args.camera_indices is None:
        args.camera_indices = _auto_detect_cameras()
        if len(args.camera_indices) < 2:
            print(f"Error: need at least 2 cameras, found {len(args.camera_indices)}")
            return 1
        print(f"Auto-detected cameras: {args.camera_indices}")

    # Landmark queue: ingress (main process) → retarget process
    landmark_queue = multiprocessing.Queue()

    def process_landmarks(landmarks):
        landmark_queue.put_nowait(landmarks)

    ingress = MultiCamIngress(
        model_path=args.model_path,
        camera_indices=args.camera_indices,
        calibration_dir=args.calibration_dir,
        marker_size_m=args.marker_size,
        marker_id=args.marker_id,
        callback=process_landmarks,
        use_reproj_filter=not args.no_reproj_filter,
        use_jitter_weight=args.jitter_weight,
        use_temporal_filter=not args.no_temporal_filter,
        debug_timing=args.debug_timing,
        use_orientation_weight=not args.no_orientation_weight,
    )

    print("Starting extrinsic calibration...")
    if not ingress.calibrate_extrinsics():
        print("Extrinsic calibration aborted.")
        ingress.cleanup()
        return 1

    # Robot control process
    robot_control_process = None
    stop_robot_control = None
    angles_queue = None

    if not args.no_robot:
        angles_queue = multiprocessing.Queue()
        stop_robot_control = multiprocessing.Event()
        robot_ready_event = multiprocessing.Event()
        robot_control_process = multiprocessing.Process(
            target=robot_control_process_worker,
            args=(angles_queue, stop_robot_control, robot_ready_event, args.model_path, args.urdf_path),
            daemon=True,
        )
        robot_control_process.start()
        if not robot_ready_event.wait(timeout=5.0):
            print("Robot initialization timeout. Exiting.")
            robot_control_process.terminate()
            ingress.cleanup()
            return 1

    # Retarget process: owns retargeter + viewer, eliminates GIL contention
    stop_retarget = multiprocessing.Event()
    viewer_stopped = multiprocessing.Event()
    retarget_process = multiprocessing.Process(
        target=retarget_process_worker,
        args=(landmark_queue, angles_queue, stop_retarget, viewer_stopped,
              args.model_path, args.urdf_path, args.retargeter,
              args.geort_checkpoint, args.geort_config,
              not args.no_viewer, args.manual_calib, args.debug_timing),
        daemon=True,
    )
    retarget_process.start()

    ingress.start()
    print("Multi-camera tracking started. Press 'q' or ESC to quit.")

    try:
        if args.no_display:
            print("Headless mode. Ctrl+C to quit.")
            while not viewer_stopped.is_set():
                time.sleep(0.1)
        else:
            while True:
                if viewer_stopped.is_set():
                    break
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), 27):
                    break
                ingress.display_frame()
    except KeyboardInterrupt:
        print("Keyboard interrupt received. Exiting...")
    finally:
        print("Stopping demo...")
        stop_retarget.set()
        ingress.cleanup()
        retarget_process.join(timeout=3.0)
        if retarget_process.is_alive():
            retarget_process.terminate()
            retarget_process.join(timeout=1.0)
        if robot_control_process and robot_control_process.is_alive():
            stop_robot_control.set()
            robot_control_process.join(timeout=3.0)
            if robot_control_process.is_alive():
                robot_control_process.terminate()
                robot_control_process.join(timeout=1.0)
    return 0


def _auto_detect_cameras(max_index=10):
    indices = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            indices.append(i)
            cap.release()
    return indices


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    sys.exit(main())
