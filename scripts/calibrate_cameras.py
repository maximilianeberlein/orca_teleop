#!/usr/bin/env python3
"""Interactive camera calibration for multi-camera triangulation.

Usage:
  # Step 1: Generate printable ChArUco board + ArUco marker
  python scripts/calibrate_cameras.py generate-board --output-dir calibration/

  # Step 2: Calibrate each camera's intrinsics (interactive)
  python scripts/calibrate_cameras.py calibrate --camera-indices 0 2 4 --output-dir calibration/ --square-size 0.035

  # Step 3: Verify ArUco extrinsic detection
  python scripts/calibrate_cameras.py verify --camera-indices 0 2 4 --calibration-dir calibration/ --marker-size 0.05
"""

import argparse
import os
import sys

import cv2
import numpy as np

from orca_teleop.orca_ingress.multicam.calibration import (
    generate_charuco_board,
    generate_aruco_marker,
    calibrate_intrinsics,
    save_intrinsics,
    load_intrinsics,
    detect_aruco_extrinsics,
    draw_aruco_axis,
)


def cmd_generate_board(args):
    os.makedirs(args.output_dir, exist_ok=True)
    board_path = os.path.join(args.output_dir, "charuco_board.png")
    marker_path = os.path.join(args.output_dir, "aruco_marker_20.png")

    generate_charuco_board(board_path)
    print(f"ChArUco board saved to: {board_path}")
    print("  Print on A4, tape to rigid surface (cardboard).")
    print("  IMPORTANT: Measure one square's side length with a ruler after printing.")

    generate_aruco_marker(20, marker_path)
    print(f"ArUco marker (ID 20) saved to: {marker_path}")
    print("  Print, cut out, tape flat on table where all cameras can see it.")
    print("  Measure the printed marker side length.")


def cmd_calibrate(args):
    os.makedirs(args.output_dir, exist_ok=True)
    marker_length = args.square_size * 0.75  # standard ChArUco ratio

    for idx in args.camera_indices:
        print(f"\n--- Calibrating camera {idx} ---")
        cam_matrix, dist_coeffs, image_size, error = calibrate_intrinsics(
            idx, args.square_size, marker_length, min_frames=args.num_frames,
        )
        out_path = os.path.join(args.output_dir, f"cam_{idx}_intrinsics.json")
        save_intrinsics(cam_matrix, dist_coeffs, image_size, out_path)
        print(f"  Saved to {out_path} (reprojection error: {error:.4f} px)")

    print("\nCalibration complete!")


def cmd_verify(args):
    cameras = []
    intrinsics = []
    for idx in args.camera_indices:
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            print(f"Warning: cannot open camera {idx}, skipping")
            continue
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cameras.append((idx, cap))

        path = os.path.join(args.calibration_dir, f"cam_{idx}_intrinsics.json")
        if not os.path.exists(path):
            print(f"Warning: no intrinsics for camera {idx} at {path}")
            cameras.pop()
            cap.release()
            continue
        intrinsics.append(load_intrinsics(path))

    if not cameras:
        print("No cameras available.")
        return 1

    print("Showing ArUco detection. Press 'q' to quit.")
    while True:
        display_frames = []
        for i, (idx, cap) in enumerate(cameras):
            ret, frame = cap.read()
            if not ret:
                continue
            cam_matrix, dist_coeffs, _ = intrinsics[i]
            result = detect_aruco_extrinsics(
                frame, cam_matrix, dist_coeffs, args.marker_size, args.marker_id,
            )
            display = frame.copy()
            if result is not None:
                rvec, tvec = result
                draw_aruco_axis(display, cam_matrix, dist_coeffs, rvec, tvec)
                cv2.putText(display, f"Cam {idx}: DETECTED", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                cv2.putText(display, f"Cam {idx}: NO MARKER", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            display_frames.append(display)

        if display_frames:
            max_h = max(f.shape[0] for f in display_frames)
            padded = []
            for f in display_frames:
                if f.shape[0] < max_h:
                    pad = np.zeros((max_h - f.shape[0], f.shape[1], 3), dtype=np.uint8)
                    f = np.vstack([f, pad])
                padded.append(f)
            cv2.imshow("ArUco Verification", np.hstack(padded))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    for _, cap in cameras:
        cap.release()
    cv2.destroyAllWindows()
    return 0


def main():
    parser = argparse.ArgumentParser(description="Camera calibration for multi-camera triangulation")
    subparsers = parser.add_subparsers(dest="command", required=True)

    gen = subparsers.add_parser("generate-board", help="Generate printable ChArUco board and ArUco marker")
    gen.add_argument("--output-dir", default="calibration/", help="Output directory (default: calibration/)")

    cal = subparsers.add_parser("calibrate", help="Calibrate camera intrinsics interactively")
    cal.add_argument("--camera-indices", type=int, nargs="+", required=True, help="Camera /dev/video indices")
    cal.add_argument("--output-dir", default="calibration/", help="Output directory for intrinsics JSON files")
    cal.add_argument("--square-size", type=float, required=True,
                     help="Measured ChArUco square side length in meters (e.g. 0.035 for 35mm)")
    cal.add_argument("--num-frames", type=int, default=20, help="Number of frames to capture (default: 20)")

    ver = subparsers.add_parser("verify", help="Verify ArUco marker detection")
    ver.add_argument("--camera-indices", type=int, nargs="+", required=True)
    ver.add_argument("--calibration-dir", default="calibration/")
    ver.add_argument("--marker-size", type=float, default=0.022, help="ArUco marker side length in meters")
    ver.add_argument("--marker-id", type=int, default=20, help="ArUco marker ID to detect (default: 20)")

    args = parser.parse_args()

    if args.command == "generate-board":
        cmd_generate_board(args)
    elif args.command == "calibrate":
        cmd_calibrate(args)
    elif args.command == "verify":
        return cmd_verify(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
