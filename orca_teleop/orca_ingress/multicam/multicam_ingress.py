import cv2
import mediapipe as mp
import numpy as np
import threading
import traceback
import time
import os
from typing import Optional, Callable, List
from orca_core import OrcaHand
from mediapipe.framework.formats import landmark_pb2
from mediapipe.python.solutions import drawing_utils, drawing_styles, hands

from .calibration import load_intrinsics, detect_aruco_extrinsics, build_projection_matrix, draw_aruco_axis
from .triangulation import triangulate_points, triangulate_with_refinement
from .one_euro_filter import OneEuroFilter

if not hasattr(mp, 'solutions'):
    import types
    mp.solutions = types.SimpleNamespace(
        drawing_utils=drawing_utils,
        drawing_styles=drawing_styles,
        hands=hands,
    )


class MultiCamIngress:

    def __init__(
        self,
        model_path: str,
        camera_indices: List[int],
        calibration_dir: str,
        marker_size_m: float = 0.05,
        marker_id: int = 20,
        callback: Optional[Callable[[np.ndarray], None]] = None,
        use_reproj_filter: bool = True,
        use_jitter_weight: bool = False,
        use_temporal_filter: bool = True,
        debug_timing: bool = False,
        use_orientation_weight: bool = True,
    ):
        self.callback = callback
        self.camera_indices = camera_indices
        self.marker_size_m = marker_size_m
        self.marker_id = marker_id
        self.n_cameras = len(camera_indices)

        self.use_reproj_filter = use_reproj_filter
        self.use_jitter_weight = use_jitter_weight
        self.use_temporal_filter = use_temporal_filter
        self.debug_timing = debug_timing
        self.use_orientation_weight = use_orientation_weight

        hand = OrcaHand(model_path)
        if hand.type not in ["left", "right"]:
            raise ValueError("hand.type must be 'left' or 'right'. Update config.yaml with type field.")
        self.hand_type = hand.type

        self.cameras = []
        self.intrinsics = []
        self.image_sizes = []
        for idx in camera_indices:
            cap = cv2.VideoCapture(idx)
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open camera {idx}")
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.cameras.append(cap)

            intrinsics_path = os.path.join(calibration_dir, f"cam_{idx}_intrinsics.json")
            if not os.path.exists(intrinsics_path):
                raise FileNotFoundError(f"Intrinsics not found: {intrinsics_path}")
            cam_matrix, dist_coeffs, image_size = load_intrinsics(intrinsics_path)
            self.intrinsics.append((cam_matrix, dist_coeffs))
            self.image_sizes.append(image_size)

        mediapipe_task_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            '..', 'mediapipe', 'hand_landmarker.task'
        )
        self.landmarkers = []
        self._cam_locks = []
        self._cam_landmarks = []
        self._cam_world_landmarks = []
        self._cam_timestamps = []
        self._cam_frames = []

        for i in range(self.n_cameras):
            lock = threading.Lock()
            self._cam_locks.append(lock)
            self._cam_landmarks.append(None)
            self._cam_world_landmarks.append(None)
            self._cam_timestamps.append(0.0)
            self._cam_frames.append(None)

            cb = self._make_result_callback(i)
            options = mp.tasks.vision.HandLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(mediapipe_task_path),
                running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
                num_hands=1,
                min_hand_detection_confidence=0.7,
                min_hand_presence_confidence=0.7,
                min_tracking_confidence=0.7,
                result_callback=cb,
            )
            self.landmarkers.append(mp.tasks.vision.HandLandmarker.create_from_options(options))

        self.projection_matrices = None
        self.running = False
        self._capture_threads = []
        self._triangulation_thread = None

        # Per-camera monotonic timestamp counters for detect_async
        self._ts_counters = [0] * self.n_cameras

        # Jitter tracking: sliding window of 2D detections per camera
        self._jitter_window_size = 8
        self._jitter_sigma_sq = 9.0  # ~3px sigma
        self._jitter_buffers = [[] for _ in range(self.n_cameras)]

        # Previous frame 3D points for NaN fallback
        self._prev_points_3d = None

        # One Euro Filter for temporal smoothing (lazy init on first valid frame)
        self._euro_filter = None

        # Debug timing accumulator
        self._timing_accum = {"collect": 0.0, "undistort": 0.0, "triangulate": 0.0, "filter": 0.0, "total": 0.0}
        self._timing_count = 0

    def _make_result_callback(self, cam_idx):
        def cb(result, _output_image, _timestamp_ms):
            try:
                if not result.hand_landmarks:
                    with self._cam_locks[cam_idx]:
                        self._cam_landmarks[cam_idx] = None
                        self._cam_world_landmarks[cam_idx] = None
                    return

                if result.handedness[0][0].category_name != self.hand_type.capitalize():
                    return

                landmarks_2d = np.array(
                    [[lm.x, lm.y] for lm in result.hand_landmarks[0]]
                )
                # Use 1.0 for all weights (like Handpose3D / aniposelib).
                # MediaPipe visibility/presence are unreliable: protobuf3
                # deserializes unset floats as 0.0, making them unusable
                # as confidence weights without per-model validation.
                confidences = np.ones(len(result.hand_landmarks[0]), dtype=np.float64)

                world_lm = None
                if self.use_orientation_weight and result.hand_world_landmarks:
                    world_lm = np.array(
                        [[lm.x, lm.y, lm.z] for lm in result.hand_world_landmarks[0]]
                    )

                with self._cam_locks[cam_idx]:
                    self._cam_landmarks[cam_idx] = (landmarks_2d, confidences)
                    self._cam_world_landmarks[cam_idx] = world_lm
                    self._cam_timestamps[cam_idx] = time.time()
            except Exception:
                traceback.print_exc()

        return cb

    def calibrate_extrinsics(self):
        print("Calibrating extrinsics — ensure ArUco marker is visible to all cameras...")
        print("Press SPACE when all cameras show detected axes, 'q' to abort.")

        while True:
            frames = []
            results = []
            for i, cap in enumerate(self.cameras):
                ret, frame = cap.read()
                if not ret:
                    frames.append(None)
                    results.append(None)
                    continue
                frames.append(frame)
                cam_matrix, dist_coeffs = self.intrinsics[i]
                result = detect_aruco_extrinsics(
                    frame, cam_matrix, dist_coeffs, self.marker_size_m, self.marker_id
                )
                results.append(result)

            display_frames = []
            detected_count = 0
            for i, frame in enumerate(frames):
                if frame is None:
                    continue
                display = frame.copy()
                if results[i] is not None:
                    rvec, tvec = results[i]
                    cam_matrix, dist_coeffs = self.intrinsics[i]
                    draw_aruco_axis(display, cam_matrix, dist_coeffs, rvec, tvec)
                    detected_count += 1
                    cv2.putText(display, "DETECTED", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                else:
                    cv2.putText(display, "NO MARKER", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.putText(display, f"Cam {self.camera_indices[i]}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                display_frames.append(display)

            if display_frames:
                max_h = max(f.shape[0] for f in display_frames)
                padded = []
                for f in display_frames:
                    if f.shape[0] < max_h:
                        pad = np.zeros((max_h - f.shape[0], f.shape[1], 3), dtype=np.uint8)
                        f = np.vstack([f, pad])
                    padded.append(f)
                tiled = np.hstack(padded)
                cv2.imshow("Extrinsic Calibration", tiled)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                cv2.destroyWindow("Extrinsic Calibration")
                return False
            elif key == ord(' ') and detected_count >= 2:
                projection_matrices = []
                for i in range(self.n_cameras):
                    if results[i] is not None:
                        rvec, tvec = results[i]
                        cam_matrix = self.intrinsics[i][0]
                        P = build_projection_matrix(cam_matrix, rvec, tvec)
                        projection_matrices.append(P)
                    else:
                        projection_matrices.append(None)
                self.projection_matrices = projection_matrices
                cv2.destroyWindow("Extrinsic Calibration")
                active = sum(1 for p in projection_matrices if p is not None)
                print(f"Extrinsic calibration done. {active}/{self.n_cameras} cameras calibrated.")
                return True

    def _capture_loop(self, cam_idx):
        cap = self.cameras[cam_idx]
        landmarker = self.landmarkers[cam_idx]
        while self.running:
            ret, frame = cap.read()
            if not ret:
                continue
            with self._cam_locks[cam_idx]:
                self._cam_frames[cam_idx] = frame.copy()
            try:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                self._ts_counters[cam_idx] += 1
                landmarker.detect_async(mp_image, self._ts_counters[cam_idx])
            except Exception:
                # MediaPipe error (e.g., dead graph runner after callback crash).
                # Keep capturing frames so display_frame still updates.
                pass

    def _compute_jitter_weights(self, cam_idx, landmarks_px):
        """Compute per-landmark jitter weights from sliding window of 2D positions."""
        buf = self._jitter_buffers[cam_idx]
        buf.append(landmarks_px.copy())
        if len(buf) > self._jitter_window_size:
            buf.pop(0)

        if len(buf) < 3:
            return np.ones(landmarks_px.shape[0], dtype=np.float64)

        stacked = np.array(buf)  # (W, N, 2)
        variance = np.mean(np.var(stacked, axis=0), axis=1)  # (N,) mean over x,y
        return 1.0 / (1.0 + variance / self._jitter_sigma_sq)

    def _compute_orientation_weight(self, world_landmarks):
        """Compute orientation confidence weight from MediaPipe world landmarks.

        Ported from mediapipe_ingress._check_orientation but returns a continuous
        weight instead of a boolean, with wider angle tolerance for multi-camera.
        """
        wrist = world_landmarks[0]
        index_mcp = world_landmarks[5]
        pinky_mcp = world_landmarks[17]
        middle_mcp = world_landmarks[9]

        if self.hand_type == "right":
            palm_normal = np.cross(index_mcp - wrist, pinky_mcp - wrist)
        else:
            palm_normal = np.cross(pinky_mcp - wrist, index_mcp - wrist)

        norm = np.linalg.norm(palm_normal)
        if norm < 1e-8:
            return 0.0
        palm_normal /= norm

        face_down_angle = np.degrees(np.arccos(np.clip(np.dot(palm_normal, [0, -1, 0]), -1.0, 1.0)))

        palm_size = np.linalg.norm(middle_mcp - wrist)
        if palm_size < 0.01:
            return 0.1

        if 70 <= face_down_angle <= 150:
            return 1.0
        elif 50 <= face_down_angle <= 170:
            return 0.3
        return 0.0

    def _triangulation_loop(self):
        while self.running:
            try:
                self._triangulation_step()
            except Exception:
                traceback.print_exc()
            time.sleep(1.0 / 90)

    def _triangulation_step(self):
        t_total = time.perf_counter()
        now = time.time()
        active_Ps = []
        active_pts = []
        active_confs = []

        t0 = time.perf_counter()
        for i in range(self.n_cameras):
            if self.projection_matrices[i] is None:
                continue
            with self._cam_locks[i]:
                data = self._cam_landmarks[i]
                world_lm = self._cam_world_landmarks[i]
                ts = self._cam_timestamps[i]

            if data is None or (now - ts) > 0.1:
                continue

            landmarks_norm, confidences = data
            conf = confidences.copy()

            if self.use_orientation_weight and world_lm is not None:
                orient_w = self._compute_orientation_weight(world_lm)
                if orient_w == 0.0:
                    continue
                conf *= orient_w

            w, h = self.image_sizes[i]
            landmarks_px = landmarks_norm.copy()
            landmarks_px[:, 0] *= w
            landmarks_px[:, 1] *= h

            cam_matrix, dist_coeffs = self.intrinsics[i]
            landmarks_px = cv2.undistortPoints(
                landmarks_px.reshape(-1, 1, 2), cam_matrix, dist_coeffs, P=cam_matrix
            ).reshape(-1, 2)

            if self.use_jitter_weight:
                jitter_w = self._compute_jitter_weights(i, landmarks_px)
                conf *= jitter_w

            active_Ps.append(self.projection_matrices[i])
            active_pts.append(landmarks_px)
            active_confs.append(conf)
        t_collect = time.perf_counter() - t0

        if len(active_Ps) < 2:
            return

        pts_2d = np.array(active_pts)
        confs = np.array(active_confs)

        t0 = time.perf_counter()
        if self.use_reproj_filter:
            points_3d, quality = triangulate_with_refinement(
                active_Ps, pts_2d, confs
            )
        else:
            points_3d = triangulate_points(active_Ps, pts_2d, confs)
        t_triangulate = time.perf_counter() - t0

        # Handle NaN landmarks (aniposelib pattern: NaN = missing)
        nan_mask = np.any(np.isnan(points_3d), axis=1)
        n_nan = int(np.sum(nan_mask))

        if n_nan > 0:
            if self._prev_points_3d is not None and n_nan <= 5:
                # Fill from previous frame (small number of missing landmarks)
                points_3d[nan_mask] = self._prev_points_3d[nan_mask]
            else:
                # Too many NaN or no previous data — drop frame
                return

        # Velocity-cap outlier rejection: clamp landmarks that jumped too far
        # (catches phantom points from temporal camera misalignment)
        if self._prev_points_3d is not None:
            delta = np.linalg.norm(points_3d - self._prev_points_3d, axis=1)
            outlier = delta > 0.04  # 4cm/frame → ~3.6 m/s at 90Hz
            points_3d[outlier] = self._prev_points_3d[outlier]

        t0 = time.perf_counter()
        # One Euro Filter for temporal smoothing (NaN-safe: see one_euro_filter.py)
        if self.use_temporal_filter:
            if self._euro_filter is None:
                self._euro_filter = OneEuroFilter(
                    min_cutoff=1.0, beta=1.5, d_cutoff=1.0
                )
            points_3d = self._euro_filter(points_3d, now)
        t_filter = time.perf_counter() - t0

        self._prev_points_3d = points_3d.copy()

        t_total_elapsed = time.perf_counter() - t_total

        if self.debug_timing:
            self._timing_accum["collect"] += t_collect
            self._timing_accum["triangulate"] += t_triangulate
            self._timing_accum["filter"] += t_filter
            self._timing_accum["total"] += t_total_elapsed
            self._timing_count += 1
            if self._timing_count >= 30:
                n = self._timing_count
                print(
                    f"[triangulation] collect={self._timing_accum['collect']/n*1000:.1f}ms "
                    f"triangulate={self._timing_accum['triangulate']/n*1000:.1f}ms "
                    f"filter={self._timing_accum['filter']/n*1000:.1f}ms "
                    f"total={self._timing_accum['total']/n*1000:.1f}ms ({n} frames avg)"
                )
                self._timing_accum = {k: 0.0 for k in self._timing_accum}
                self._timing_count = 0

        if self.callback and not np.any(np.isnan(points_3d)):
            self.callback(points_3d)

    def start(self):
        if self.projection_matrices is None:
            raise RuntimeError("Must call calibrate_extrinsics() before start()")
        if self.running:
            return
        self.running = True
        for i in range(self.n_cameras):
            t = threading.Thread(target=self._capture_loop, args=(i,), daemon=True)
            t.start()
            self._capture_threads.append(t)
        self._triangulation_thread = threading.Thread(target=self._triangulation_loop, daemon=True)
        self._triangulation_thread.start()

    def stop(self):
        self.running = False
        for t in self._capture_threads:
            t.join(timeout=2.0)
        if self._triangulation_thread:
            self._triangulation_thread.join(timeout=2.0)
        self._capture_threads.clear()

    def display_frame(self):
        frames = []
        for i in range(self.n_cameras):
            with self._cam_locks[i]:
                frame = self._cam_frames[i]
                data = self._cam_landmarks[i]
            if frame is None:
                continue
            display = frame.copy()

            if data is not None:
                landmarks_norm, _ = data
                h, w = display.shape[:2]
                proto = landmark_pb2.NormalizedLandmarkList()
                for lm in landmarks_norm:
                    new_lm = proto.landmark.add()
                    new_lm.x, new_lm.y, new_lm.z = float(lm[0]), float(lm[1]), 0.0
                mp.solutions.drawing_utils.draw_landmarks(
                    display, proto, mp.solutions.hands.HAND_CONNECTIONS,
                    mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                    mp.solutions.drawing_styles.get_default_hand_connections_style(),
                )

            cv2.putText(display, f"Cam {self.camera_indices[i]}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            frames.append(display)

        if not frames:
            return

        max_h = max(f.shape[0] for f in frames)
        padded = []
        for f in frames:
            if f.shape[0] < max_h:
                pad = np.zeros((max_h - f.shape[0], f.shape[1], 3), dtype=np.uint8)
                f = np.vstack([f, pad])
            padded.append(f)
        tiled = np.hstack(padded)
        cv2.imshow("Multi-Camera Hand Tracking", tiled)

    def cleanup(self):
        self.stop()
        for cap in self.cameras:
            cap.release()
        for lm in self.landmarkers:
            lm.close()
        cv2.destroyAllWindows()
