import os
import sys
import json
from typing import Dict, Union
import numpy as np
import pytorch_kinematics as pk
import torch
from orca_core import OrcaHand
from .utils import retargeter_utils
from .utils.manual_calibration import apply_manual_calibration

# Add GeoRT to path so we can import its modules
_GEORT_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "third_party", "GeoRT")
if _GEORT_ROOT not in sys.path:
    sys.path.insert(0, _GEORT_ROOT)

from geort.model import IKModel


class NeuralGeoRTRetargeter:
    """Neural retargeter using GeoRT's pre-trained IK model (single MLP forward pass)."""

    def __init__(self, model_path: Union[OrcaHand, str] = None, urdf_path: Union[str, None] = None,
                 geort_checkpoint: str = None, geort_config: str = None, source: str = "none") -> None:

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.source = source
        self.target_angles = None
        self.mano_points = None

        # Load ORCA hand for joint IDs and ROM limits
        hand = OrcaHand(model_path)
        if hand.type not in ["left", "right"]:
            raise ValueError("hand.type must be 'left' or 'right'. Update config.yaml with type field.")
        self.hand_type = hand.type
        self.joint_ids = hand.joint_ids
        self.urdf_joint_ids = [f"{hand.type}_{joint_id}" for joint_id in self.joint_ids]
        lower_limits, upper_limits = map(list, zip(*hand.joint_roms_dict.values()))
        self.wrist_limit_lower = lower_limits[16]
        self.wrist_limit_upper = upper_limits[16]

        # Load GeoRT config
        with open(geort_config, 'r') as f:
            config = json.load(f)

        # Parse keypoint info from config (same logic as geort.utils.config_utils)
        joint_order = config["joint_order"]
        keypoint_joints = []
        self.human_ids = []
        for info in config["fingertip_link"]:
            self.human_ids.append(info["human_hand_id"])
            keypoint_joints.append([joint_order.index(j) for j in info["joint"]])

        # Joint limits from trained config (added during training)
        joint_lower = np.array(config["joint"]["lower"])
        joint_upper = np.array(config["joint"]["upper"])
        self.joint_lower = joint_lower
        self.joint_upper = joint_upper

        # Build IK model and load checkpoint
        self.ik_model = IKModel(keypoint_joints=keypoint_joints).to(self.device)
        state_dict = torch.load(geort_checkpoint, map_location=self.device, weights_only=True)
        self.ik_model.load_state_dict(state_dict)
        self.ik_model.eval()

        # Build mapping from GeoRT's joint_order (16 joints) to ORCA's urdf_joint_ids (17 joints, last is wrist)
        # GeoRT outputs joints in config's joint_order; we need to reorder to match self.urdf_joint_ids
        self.geort_to_orca_indices = []
        for geort_idx, geort_joint_name in enumerate(joint_order):
            # Strip hand prefix to get the bare joint id (e.g. "right_thumb_abd" -> "thumb_abd")
            bare_name = geort_joint_name.split("_", 1)[1] if geort_joint_name.startswith(("left_", "right_")) else geort_joint_name
            orca_idx = self.joint_ids.index(bare_name)
            self.geort_to_orca_indices.append(orca_idx)

        # Source-specific landmark indices for extracting from canonical frame
        finger_names = [info["name"] for info in config["fingertip_link"]]
        self._fingertip_indices = self._get_fingertip_indices(finger_names)
        # Rotation from GeoRT canonical frame to URDF palm-local frame (90° around Z).
        # Canonical: X=palm normal, Y=across palm (index→ring), Z=along fingers
        # Palm:      X=across palm (thumb→pinky), Y=palm normal, Z=along fingers
        self._canonical_to_palm = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=np.float32)

        # --- Visualization: URDF world-frame transform (same approach as default Retargeter) ---
        self._fingers = ["thumb", "index", "middle", "ring", "pinky"]
        with open(urdf_path, 'r') as f:
            viz_chain = pk.build_chain_from_urdf(f.read())
        viz_root = torch.zeros(1, 3)
        self._urdfhand_center, self._urdfhand_rot_matrix, viz_frames = retargeter_utils.get_urdf_model_params(
            viz_chain, self.hand_type, self._fingers, viz_root)
        viz_fingertip_offsets = retargeter_utils.get_fingertip_offset_tensors(self._fingers, "cpu")
        viz_joint_names = viz_chain.get_joint_parameter_names()
        viz_reorder = [viz_joint_names.index(f"{self.hand_type}_{jid}") for jid in self.joint_ids]
        # Use halfway between curled (0) and extended (-ref) to approximate relaxed pose
        neutral_angles = torch.zeros(viz_chain.n_joints)
        ref_rad = torch.tensor(
            retargeter_utils.get_ref_offsets_array(self.joint_ids), dtype=torch.float32)
        neutral_angles[viz_reorder] = -0.5 * ref_rad
        urdf_ft, urdf_palm = retargeter_utils.extract_orca_fingertips_and_palm(
            viz_chain, neutral_angles, viz_frames, self.hand_type, self._fingers, viz_root,
            fingertip_offsets=viz_fingertip_offsets)
        urdf_kvs = retargeter_utils.get_keyvectors(urdf_ft, urdf_palm)
        self._urdf_kv_mags = 0.9 * np.array([kv.detach().cpu().norm().item() for kv in urdf_kvs])
        self._mano_scale = 1.0
        self._cal_mags = []
        self._cal_frames = 30

        # Manual calibration (adjustable at runtime via viewer sliders, viz only for neural IK)
        self.manual_scale = 1.0
        self.manual_rotation = np.zeros(3)
        self.manual_translation = np.zeros(3)

    def _get_fingertip_indices(self, finger_names):
        """Get fingertip landmark indices per source, in config's finger order."""
        mediapipe_tips = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}
        manus_tips = {"thumb": 24, "index": 4, "middle": 9, "ring": 19, "pinky": 14}
        tip_map = manus_tips if self.source == "manus" else mediapipe_tips
        return [tip_map[name] for name in finger_names]

    def retarget(self, data: np.ndarray, manual_wrist_angle: Union[float, None] = None) -> Dict[str, float]:

        if self.source == "avp":
            joints, computed_wrist_angle = retargeter_utils.preprocess_avp_data(data, self.hand_type)
        elif self.source == "mediapipe":
            joints, computed_wrist_angle = retargeter_utils.preprocess_mediapipe_data(data)
        elif self.source == "manus":
            joints, computed_wrist_angle = retargeter_utils.preprocess_manus_data(data)
        else:
            raise ValueError(f"Unsupported source: {self.source}")

        final_wrist_angle = manual_wrist_angle if manual_wrist_angle is not None else computed_wrist_angle

        # Transform to GeoRT's canonical wrist-centered frame, then to URDF palm frame
        canonical = retargeter_utils.to_geort_canonical_frame(joints, self.source)
        palm_frame = canonical @ self._canonical_to_palm

        # Extract fingertip positions
        ik_input = palm_frame[self._fingertip_indices]  # (N_fingers, 3)

        # IK model forward pass
        with torch.no_grad():
            ik_input_t = torch.from_numpy(ik_input).unsqueeze(0).float().to(self.device)
            normalized_angles = self.ik_model(ik_input_t)  # (1, 16) in [-1, 1]
            normalized_angles = normalized_angles[0].cpu().numpy()  # (16,)

        # Unnormalize: [-1, 1] → [lower, upper] (GeoRT convention: tanh output)
        raw_angles = (normalized_angles / 2.0 + 0.5) * (self.joint_upper - self.joint_lower) + self.joint_lower

        # GeoRT outputs radians (URDF convention). ORCA retargeters work in degrees internally.
        angles_deg = np.rad2deg(raw_angles)

        # Reorder from GeoRT's joint_order to ORCA's joint_ids order
        orca_angles = np.zeros(len(self.urdf_joint_ids))
        for geort_idx, orca_idx in enumerate(self.geort_to_orca_indices):
            orca_angles[orca_idx] = angles_deg[geort_idx]

        # Wrist angle (handled separately, not part of GeoRT model)
        final_wrist_angle = np.clip(final_wrist_angle, self.wrist_limit_lower, self.wrist_limit_upper)
        orca_angles[-1] = final_wrist_angle if self.hand_type == "left" else -final_wrist_angle
        self.target_angles = orca_angles

        # --- Visualization: transform mano points to URDF world frame (same as default Retargeter) ---
        mano_viz = retargeter_utils.get_normalized_local_manohand_joint_pos(joints, self.source)
        if len(self._cal_mags) < self._cal_frames:
            mano_t = torch.from_numpy(mano_viz).to(self.device)
            mano_ft, mano_palm = retargeter_utils.extract_mano_fingertips_and_palm(mano_t, self._fingers, self.source)
            mano_kvs = retargeter_utils.get_keyvectors(mano_ft, mano_palm)
            mano_mags = np.array([kv.detach().cpu().norm().item() for kv in mano_kvs])
            self._cal_mags.append(mano_mags)
            if len(self._cal_mags) == self._cal_frames:
                all_mags = np.array(self._cal_mags)
                median_mano = np.median(all_mags, axis=0)
                ratios = self._urdf_kv_mags / np.clip(median_mano, 1e-6, None)
                self._mano_scale = float(np.median(ratios))
                print(f"Auto-scale calibrated: {self._mano_scale:.4f}")
        mano_viz = mano_viz * self._mano_scale
        mano_viz = mano_viz @ self._urdfhand_rot_matrix.T + self._urdfhand_center + np.array([0, 0, -0.02])
        mano_viz = apply_manual_calibration(
            mano_viz, self._urdfhand_center,
            self.manual_scale, self.manual_rotation, self.manual_translation)
        self.mano_points = retargeter_utils.rotate_points_around_y(mano_viz, final_wrist_angle, self.source, self.hand_type)

        return {urdf_joint_id: np.deg2rad(angle) for urdf_joint_id, angle in zip(self.urdf_joint_ids, orca_angles)}
