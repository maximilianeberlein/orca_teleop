import sys
import time
import multiprocessing
import cv2
import argparse
from orca_teleop import MediaPipeIngress, Retargeter

debug_timing = False
_timing_accum = {"retarget": 0.0, "viewer": 0.0, "total": 0.0}
_timing_count = 0


def robot_control_process_worker(q, stop, ready, model_path, urdf_path=None):
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
        while not stop.is_set():
            try:
                angles = q.get(timeout=0.1)
                while not q.empty():
                    try:
                        angles = q.get_nowait()
                    except Exception:
                        break
                if angles:
                    hand.set_joint_pos(urdf_angles_to_physical(angles, ref_offsets))
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


def process_landmarks(landmarks):
    global _timing_accum, _timing_count
    t_total = time.perf_counter()

    t0 = time.perf_counter()
    angles = retargeter.retarget({"hand_landmarks": landmarks})
    t_retarget = time.perf_counter() - t0

    if angles_queue:
        angles_queue.put_nowait(angles)

    t0 = time.perf_counter()
    if viewer:
        viewer.update(angles)
        if show_mano and retargeter.mano_points is not None:
            viewer.update_mano_points(retargeter.mano_points)
        if hasattr(retargeter, 'fk_points') and retargeter.fk_points is not None:
            viewer.update_fk_points(retargeter.fk_points)
    t_viewer = time.perf_counter() - t0

    t_total_elapsed = time.perf_counter() - t_total

    if debug_timing:
        _timing_accum["retarget"] += t_retarget
        _timing_accum["viewer"] += t_viewer
        _timing_accum["total"] += t_total_elapsed
        _timing_count += 1
        if _timing_count >= 30:
            n = _timing_count
            print(
                f"[retarget] retarget={_timing_accum['retarget']/n*1000:.1f}ms "
                f"viewer={_timing_accum['viewer']/n*1000:.1f}ms "
                f"total={_timing_accum['total']/n*1000:.1f}ms ({n} frames avg)"
            )
            _timing_accum = {k: 0.0 for k in _timing_accum}
            _timing_count = 0


def main():
    global retargeter, angles_queue, viewer, show_mano, debug_timing
    parser = argparse.ArgumentParser(description='MediaPipe to Orca Hand teleop demo')
    parser.add_argument('model_path')
    parser.add_argument('urdf_path')
    parser.add_argument('--no-display', action='store_true')
    parser.add_argument('--no-viewer', action='store_true', help='Disable 3D URDF viewer')
    parser.add_argument('--no-robot', action='store_true', help='Run without robot hardware')
    parser.add_argument('--show-mano', action='store_true', help='Show MANO landmarks in 3D viewer')
    parser.add_argument('--manual-calib', action='store_true', help='Enable manual calibration sliders in viewer')
    parser.add_argument('--retargeter', choices=['default', 'absolute', 'geort', 'neural-geort', 'pyroki'], default='default', help='Retargeter to use')
    parser.add_argument('--geort-checkpoint', type=str, default=None, help='Path to GeoRT IK model checkpoint (.pth)')
    parser.add_argument('--geort-config', type=str, default=None, help='Path to GeoRT config JSON (with joint limits)')
    parser.add_argument('--debug-timing', action='store_true', help='Print per-stage timing breakdown every 30 frames')
    args = parser.parse_args()
    show_mano = args.show_mano or args.manual_calib
    debug_timing = args.debug_timing

    viewer = None
    if not args.no_viewer:
        try:
            from orca_teleop.viewer import URDFViewer
            viewer = URDFViewer(args.urdf_path)
        except ImportError:
            print("viser not installed — skipping 3D viewer (pip install -e '.[viewer]')")
        except Exception as e:
            print(f"Failed to start viewer: {e}")

    if args.retargeter == 'neural-geort':
        if not args.geort_checkpoint or not args.geort_config:
            print("Error: --geort-checkpoint and --geort-config required for neural-geort retargeter")
            return 1
        from orca_teleop import NeuralGeoRTRetargeter
        retargeter = NeuralGeoRTRetargeter(args.model_path, args.urdf_path,
                                           geort_checkpoint=args.geort_checkpoint,
                                           geort_config=args.geort_config, source="mediapipe")
    elif args.retargeter == 'absolute':
        from orca_teleop import AbsoluteRetargeter
        retargeter = AbsoluteRetargeter(args.model_path, args.urdf_path, source="mediapipe")
    elif args.retargeter == 'geort':
        from orca_teleop import GeoRTRetargeter
        retargeter = GeoRTRetargeter(args.model_path, args.urdf_path, source="mediapipe")
    elif args.retargeter == 'pyroki':
        from orca_teleop import PyRoKIRetargeter
        retargeter = PyRoKIRetargeter(args.model_path, args.urdf_path, source="mediapipe")
    else:
        retargeter = Retargeter(args.model_path, args.urdf_path, source="mediapipe")

    if args.manual_calib and viewer:
        viewer.add_calibration_controls(retargeter)

    ingress = MediaPipeIngress(args.model_path, callback=process_landmarks)

    robot_control_process = None
    stop_robot_control = None
    angles_queue = None

    if not args.no_robot:
        from orca_core import OrcaHand
        angles_queue = multiprocessing.Queue()
        stop_robot_control = multiprocessing.Event()
        robot_ready_event = multiprocessing.Event()
        robot_control_process = multiprocessing.Process(
            target=robot_control_process_worker,
            args=(angles_queue, stop_robot_control, robot_ready_event, args.model_path, args.urdf_path), daemon=True)

        robot_control_process.start()
        if not robot_ready_event.wait(timeout=5.0):
            print("Robot initialization timeout. Exiting.")
            robot_control_process.terminate()
            return 1

    ingress.start()

    try:
        if args.no_display:
            print("Headless mode. Ctrl+C to quit")
            while not (viewer and viewer.stopped):
                time.sleep(0.1)
        else:
            print("Demo running. Press 'q' or ESC to quit")
            while True:
                if viewer and viewer.stopped:
                    break
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), 27):
                    break
                ingress.display_frame()
    except KeyboardInterrupt:
        print("Keyboard interrupt received. Exiting...")
    finally:
        print("Stopping demo...")
        ingress.cleanup()
        if viewer:
            viewer.close()
        if robot_control_process and robot_control_process.is_alive():
            stop_robot_control.set()
            robot_control_process.join(timeout=3.0)
            if robot_control_process.is_alive():
                robot_control_process.terminate()
                robot_control_process.join(timeout=1.0)
    return 0


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    sys.exit(main())
