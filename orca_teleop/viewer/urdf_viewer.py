import re
import time
import webbrowser
from pathlib import Path
from functools import partial

import numpy as np
import viser
from viser.extras import ViserUrdf
import yourdfpy


FINGER_COLORS = np.array([
    [200, 200, 200],  # 0: wrist
    [255, 80, 80],    # 1-4: thumb
    [255, 80, 80],
    [255, 80, 80],
    [255, 80, 80],
    [80, 255, 80],    # 5-8: index
    [80, 255, 80],
    [80, 255, 80],
    [80, 255, 80],
    [80, 80, 255],    # 9-12: middle
    [80, 80, 255],
    [80, 80, 255],
    [80, 80, 255],
    [255, 255, 80],   # 13-16: ring
    [255, 255, 80],
    [255, 255, 80],
    [255, 255, 80],
    [255, 80, 255],   # 17-20: pinky
    [255, 80, 255],
    [255, 80, 255],
    [255, 80, 255],
], dtype=np.uint8)

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]

# Remap Manus 25 nodes to 21-node MediaPipe-like layout (drop unused CMC nodes 5, 10, 15, 20)
# Reordered to: wrist, thumb(21-24), index(1-4), middle(6-9), ring(16-19), pinky(11-14)
MANUS_TO_MEDIAPIPE = [0, 21, 22, 23, 24, 1, 2, 3, 4, 6, 7, 8, 9, 16, 17, 18, 19, 11, 12, 13, 14]


def _package_uri_handler(fname, dir):
    stripped = re.sub(r"^package://[^/]*/", "", fname)
    resolved = Path(dir) / stripped
    if resolved.exists():
        return str(resolved)
    return yourdfpy.filename_handler_magic(fname, dir=dir)


class URDFViewer:
    def __init__(self, urdf_path, port=8080):
        self._server = viser.ViserServer(port=port)
        urdf_dir = Path(urdf_path).parent
        urdf = yourdfpy.URDF.load(
            urdf_path,
            filename_handler=partial(_package_uri_handler, dir=str(urdf_dir)),
            build_scene_graph=True,
            load_meshes=True,
            load_collision_meshes=False,
        )
        self._viser_urdf = ViserUrdf(self._server, urdf_or_path=urdf)
        self._joint_names = self._viser_urdf.get_actuated_joint_names()
        self._viser_urdf.update_cfg(np.zeros(len(self._joint_names)))
        self._server.scene.add_grid("/ground", width=2.0, height=2.0)

        self._mano_visible = self._server.gui.add_checkbox(
            "Show MANO Points", initial_value=True,
        )
        self._fk_visible = self._server.gui.add_checkbox(
            "Show FK Points", initial_value=False,
        )
        self._stop_button = self._server.gui.add_button("Stop")

        self._stopped = False
        self._has_connected = False
        self._start_time = time.time()
        self._mano_points_handle = None
        self._mano_lines_handle = None
        self._fk_points_handle = None
        self._fk_lines_handle = None

        @self._stop_button.on_click
        def _(_):
            self._stopped = True

        @self._server.on_client_connect
        def _(client):
            self._has_connected = True

        @self._server.on_client_disconnect
        def _(client):
            if self._has_connected and len(self._server.get_clients()) == 0 and time.time() - self._start_time > 3.0:
                self._stopped = True

        webbrowser.open(f"http://localhost:{port}")

    @property
    def stopped(self):
        return self._stopped

    def update(self, joint_angles):
        cfg = np.array([joint_angles.get(name, 0.0) for name in self._joint_names])
        self._viser_urdf.update_cfg(cfg)

    def update_mano_points(self, points):
        if not self._mano_visible.value:
            if self._mano_points_handle is not None:
                self._mano_points_handle.remove()
                self._mano_points_handle = None
            if self._mano_lines_handle is not None:
                self._mano_lines_handle.remove()
                self._mano_lines_handle = None
            return

        points = points.astype(np.float32)
        if len(points) == 25:
            points = points[MANUS_TO_MEDIAPIPE]
        n = len(points)
        colors = FINGER_COLORS[:n]
        connections = HAND_CONNECTIONS

        self._mano_points_handle = self._server.scene.add_point_cloud(
            "/mano_points",
            points=points,
            colors=colors,
            point_size=0.005,
        )

        line_points = np.array(
            [[points[i], points[j]] for i, j in connections if i < n and j < n],
            dtype=np.float32,
        )
        line_colors = np.full((len(line_points), 2, 3), 150, dtype=np.uint8)
        self._mano_lines_handle = self._server.scene.add_line_segments(
            "/mano_lines",
            points=line_points,
            colors=line_colors,
        )

    def update_fk_points(self, points):
        if not self._fk_visible.value:
            if self._fk_points_handle is not None:
                self._fk_points_handle.remove()
                self._fk_points_handle = None
            if self._fk_lines_handle is not None:
                self._fk_lines_handle.remove()
                self._fk_lines_handle = None
            return

        # points: (6, 3) = [thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip, palm]
        points = points.astype(np.float32)
        colors = np.array([
            [255, 0, 0],      # thumb: red
            [0, 255, 0],      # index: green
            [0, 0, 255],      # middle: blue
            [255, 255, 0],    # ring: yellow
            [255, 0, 255],    # pinky: magenta
            [255, 255, 255],  # palm: white
        ], dtype=np.uint8)

        self._fk_points_handle = self._server.scene.add_point_cloud(
            "/fk_points",
            points=points,
            colors=colors,
            point_size=0.008,
        )

        # Draw lines from palm (index 5) to each fingertip (indices 0-4)
        palm = points[5]
        line_points = np.array([[palm, points[i]] for i in range(5)], dtype=np.float32)
        line_colors = np.array([
            [[255, 0, 0], [255, 0, 0]],
            [[0, 255, 0], [0, 255, 0]],
            [[0, 0, 255], [0, 0, 255]],
            [[255, 255, 0], [255, 255, 0]],
            [[255, 0, 255], [255, 0, 255]],
        ], dtype=np.uint8)
        self._fk_lines_handle = self._server.scene.add_line_segments(
            "/fk_lines",
            points=line_points,
            colors=line_colors,
        )

    def add_calibration_controls(self, retargeter):
        with self._server.gui.add_folder("Manual Calibration", expand_by_default=False) as folder:
            tx = self._server.gui.add_slider("Translate X (m)", -0.05, 0.05, 0.001, 0.0)
            ty = self._server.gui.add_slider("Translate Y (m)", -0.05, 0.05, 0.001, 0.0)
            tz = self._server.gui.add_slider("Translate Z (m)", -0.05, 0.05, 0.001, 0.0)
            rx = self._server.gui.add_slider("Roll (deg)", -30.0, 30.0, 0.5, 0.0)
            ry = self._server.gui.add_slider("Pitch (deg)", -30.0, 30.0, 0.5, 0.0)
            rz = self._server.gui.add_slider("Yaw (deg)", -30.0, 30.0, 0.5, 0.0)
            scale = self._server.gui.add_slider("Scale", 0.5, 2.0, 0.01, 1.0)
            reset_btn = self._server.gui.add_button("Reset")
            print_btn = self._server.gui.add_button("Print Values")

        def _sync_translation(_=None):
            retargeter.manual_translation = np.array([tx.value, ty.value, tz.value])

        def _sync_rotation(_=None):
            retargeter.manual_rotation = np.array([rx.value, ry.value, rz.value])

        def _sync_scale(_=None):
            retargeter.manual_scale = scale.value

        tx.on_update(_sync_translation)
        ty.on_update(_sync_translation)
        tz.on_update(_sync_translation)
        rx.on_update(_sync_rotation)
        ry.on_update(_sync_rotation)
        rz.on_update(_sync_rotation)
        scale.on_update(_sync_scale)

        @reset_btn.on_click
        def _(_):
            tx.value = ty.value = tz.value = 0.0
            rx.value = ry.value = rz.value = 0.0
            scale.value = 1.0
            _sync_translation()
            _sync_rotation()
            _sync_scale()

        @print_btn.on_click
        def _(_):
            print(f"Manual calibration values:")
            print(f"  translation: [{tx.value:.4f}, {ty.value:.4f}, {tz.value:.4f}]")
            print(f"  rotation:    [{rx.value:.1f}, {ry.value:.1f}, {rz.value:.1f}]")
            print(f"  scale:       {scale.value:.3f}")

    def close(self):
        self._server.stop()
