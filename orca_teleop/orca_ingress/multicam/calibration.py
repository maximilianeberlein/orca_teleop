import json
import numpy as np
import cv2


CHARUCO_DICT = cv2.aruco.DICT_6X6_250
CHARUCO_SQUARES_X = 7
CHARUCO_SQUARES_Y = 5


def generate_charuco_board(output_path, square_length_mm=40, marker_length_mm=30):
    dictionary = cv2.aruco.getPredefinedDictionary(CHARUCO_DICT)
    board = cv2.aruco.CharucoBoard(
        (CHARUCO_SQUARES_X, CHARUCO_SQUARES_Y),
        square_length_mm / 1000.0,
        marker_length_mm / 1000.0,
        dictionary,
    )
    img = board.generateImage((2100, 1500))  # ~A4 at 250 DPI
    cv2.imwrite(output_path, img)
    return output_path


def generate_aruco_marker(marker_id, output_path, size_px=500):
    dictionary = cv2.aruco.getPredefinedDictionary(CHARUCO_DICT)
    img = cv2.aruco.generateImageMarker(dictionary, marker_id, size_px)
    cv2.imwrite(output_path, img)
    return output_path


def calibrate_intrinsics(camera_index, square_length_m, marker_length_m, min_frames=15):
    dictionary = cv2.aruco.getPredefinedDictionary(CHARUCO_DICT)
    board = cv2.aruco.CharucoBoard(
        (CHARUCO_SQUARES_X, CHARUCO_SQUARES_Y),
        square_length_m,
        marker_length_m,
        dictionary,
    )
    charuco_detector = cv2.aruco.CharucoDetector(board)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {camera_index}")
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    all_charuco_corners = []
    all_charuco_ids = []
    image_size = None

    print(f"Camera {camera_index}: show ChArUco board, press SPACE to capture ({min_frames} needed), 'q' to finish early")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        if image_size is None:
            image_size = (frame.shape[1], frame.shape[0])

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        charuco_corners, charuco_ids, marker_corners, marker_ids = charuco_detector.detectBoard(gray)

        display = frame.copy()
        if marker_ids is not None and len(marker_ids) > 0:
            cv2.aruco.drawDetectedMarkers(display, marker_corners, marker_ids)
        if charuco_corners is not None and len(charuco_corners) >= 4:
            cv2.aruco.drawDetectedCornersCharuco(display, charuco_corners, charuco_ids)

        cv2.putText(display, f"Captured: {len(all_charuco_corners)}/{min_frames}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow(f"Calibration - Camera {camera_index}", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            if charuco_corners is not None and len(charuco_corners) >= 4:
                all_charuco_corners.append(charuco_corners)
                all_charuco_ids.append(charuco_ids)
                print(f"  Captured frame {len(all_charuco_corners)} ({len(charuco_corners)} corners)")
            else:
                print("  Not enough corners detected, try again")

        if len(all_charuco_corners) >= min_frames:
            print(f"Collected {min_frames} frames, calibrating...")
            break

    cap.release()
    cv2.destroyWindow(f"Calibration - Camera {camera_index}")

    if len(all_charuco_corners) < 5:
        raise RuntimeError(f"Only {len(all_charuco_corners)} frames captured, need at least 5")

    obj_points = []
    for ids in all_charuco_ids:
        obj_pts = board.getChessboardCorners()[ids.flatten()]
        obj_points.append(obj_pts.reshape(-1, 1, 3))

    img_points = [c.reshape(-1, 1, 2) for c in all_charuco_corners]

    retval, camera_matrix, dist_coeffs, _, _ = cv2.calibrateCamera(
        obj_points, img_points, image_size, None, None
    )
    print(f"Camera {camera_index}: reprojection error = {retval:.4f} px")
    return camera_matrix, dist_coeffs, image_size, retval


def save_intrinsics(camera_matrix, dist_coeffs, image_size, path):
    data = {
        "camera_matrix": camera_matrix.tolist(),
        "dist_coeffs": dist_coeffs.tolist(),
        "image_size": list(image_size),
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_intrinsics(path):
    with open(path, "r") as f:
        data = json.load(f)
    camera_matrix = np.array(data["camera_matrix"], dtype=np.float64)
    dist_coeffs = np.array(data["dist_coeffs"], dtype=np.float64)
    image_size = tuple(data["image_size"])
    return camera_matrix, dist_coeffs, image_size


def detect_aruco_extrinsics(frame, camera_matrix, dist_coeffs, marker_size_m, marker_id=20):
    dictionary = cv2.aruco.getPredefinedDictionary(CHARUCO_DICT)
    detector_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, detector_params)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    marker_corners, marker_ids, _ = detector.detectMarkers(gray)

    if marker_ids is None:
        return None

    target_idx = None
    for i, mid in enumerate(marker_ids.flatten()):
        if mid == marker_id:
            target_idx = i
            break
    if target_idx is None:
        return None

    corners = marker_corners[target_idx]
    obj_points = np.array([
        [-marker_size_m / 2, marker_size_m / 2, 0],
        [marker_size_m / 2, marker_size_m / 2, 0],
        [marker_size_m / 2, -marker_size_m / 2, 0],
        [-marker_size_m / 2, -marker_size_m / 2, 0],
    ], dtype=np.float64)

    success, rvec, tvec = cv2.solvePnP(obj_points, corners.reshape(4, 2), camera_matrix, dist_coeffs)
    if not success:
        return None

    return rvec, tvec


def build_projection_matrix(camera_matrix, rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    Rt = np.hstack([R, tvec.reshape(3, 1)])
    P = camera_matrix @ Rt
    return P


def draw_aruco_axis(frame, camera_matrix, dist_coeffs, rvec, tvec, length=0.03):
    cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, length)
    return frame
