import os
from typing import Dict, Tuple, Union
import numpy as np
import pytorch_kinematics as pk
import torch
import yaml
from orca_core import OrcaHand
from .utils import retargeter_utils
from .utils.manual_calibration import apply_manual_calibration
from .utils.urdf_renamer import ensure_semantic_urdf


def _build_loss_fn(chain, joint_reorder_indices, optimization_frames,
                   hand_type, fingers, root, fingertip_offsets,
                   loss_coeffs, use_scalar_distance,
                   regularizer_weights, regularizer_zeros):
    """Build a loss closure with pre-resolved frame names and offsets.

    All chain/config parameters are captured as closure constants to reduce
    per-call Python overhead (dict lookups, function calls, etc.).
    Returns a callable: (orcahand_joint_angles, mano_tips_stacked, mano_palm) -> scalar loss
    """
    n_joints = chain.n_joints
    deg2rad = np.pi / 180.0
    palm_offset = torch.tensor([0, 0, 0.015], device=root.device)

    fingertip_names = [retargeter_utils.get_fingertip_urdf_name(hand_type, f) for f in fingers]
    thumb_base_name = retargeter_utils.get_finger_base_urdf_name(hand_type, "thumb")
    pinky_base_name = retargeter_utils.get_finger_base_urdf_name(hand_type, "pinky")
    tip_offsets = [fingertip_offsets[f] for f in fingers]
    scalar_mask = use_scalar_distance

    def loss_fn(gc_joints, mano_tips_stacked, mano_palm):
        angles = torch.zeros(n_joints, device=gc_joints.device)
        angles[joint_reorder_indices] = gc_joints * deg2rad
        transforms = chain.forward_kinematics(angles, frame_indices=optimization_frames)

        tips = [transforms[name].transform_points(offset) for name, offset in zip(fingertip_names, tip_offsets)]

        thumb_base = transforms[thumb_base_name].transform_points(root)
        pinky_base = transforms[pinky_base_name].transform_points(root)
        palm = torch.mean(torch.cat([thumb_base, pinky_base], dim=0), dim=0, keepdim=True) - palm_offset

        loss = torch.zeros(1, device=gc_joints.device)
        for i in range(5):
            kv_urdf = tips[i] - palm
            kv_mano = mano_tips_stacked[i] - mano_palm
            if scalar_mask[i]:
                loss = loss + loss_coeffs[i] * (torch.norm(kv_mano) - torch.norm(kv_urdf)) ** 2
            else:
                loss = loss + loss_coeffs[i] * torch.norm(kv_mano - kv_urdf) ** 2

        loss = loss + torch.sum(regularizer_weights * (gc_joints - regularizer_zeros) ** 2)
        return loss

    return loss_fn


class Retargeter:
    """Retargeter class for Orca Hand to retarget MANO joint angles to Orca Hand joint angles."""
    
    def __init__(self, model_path: Union[OrcaHand, str] = None, urdf_path: Union[str, None] = None, source: str = "none", verbose: bool = False) -> None:

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.source = source
        self.verbose = verbose
        self.target_angles = None
        self.mano_points = None

        hand = OrcaHand(model_path)
        if hand.type not in ["left", "right"]:
            raise ValueError("hand.type must be 'left' or 'right'. Update config.yaml with type field.")
        self.hand_type = hand.type
        self.joint_ids = hand.joint_ids
        self.urdf_joint_ids = [f"{hand.type}_{joint_id}" for joint_id in self.joint_ids]

        if not os.path.exists(urdf_path):
            raise ValueError(f"URDF file not found at {urdf_path}")
        self._ref_offsets = ensure_semantic_urdf(urdf_path, self.hand_type, self.joint_ids)
        with open(urdf_path, 'r') as f:
            self.chain = pk.build_chain_from_urdf(f.read()).to(device=self.device)
        self.fingers = ["thumb", "index", "middle", "ring", "pinky"]
        lower_limits, upper_limits = map(list, zip(*hand.joint_roms_dict.values()))
        self.wrist_idx = self.joint_ids.index("wrist")
        self.wrist_limit_lower = lower_limits[self.wrist_idx]
        self.wrist_limit_upper = upper_limits[self.wrist_idx]
        lower_limits[self.wrist_idx] = upper_limits[self.wrist_idx] = 0.0  # Keep wrist constrained to zero during optimization
        ref_offsets_deg = np.rad2deg(retargeter_utils.get_ref_offsets_array(self.joint_ids, self._ref_offsets))
        lower_limits_urdf = np.array(lower_limits) - ref_offsets_deg
        upper_limits_urdf = np.array(upper_limits) - ref_offsets_deg
        self.joint_angle_limits_lower = torch.tensor(lower_limits_urdf, device=self.device)
        self.joint_angle_limits_upper = torch.tensor(upper_limits_urdf, device=self.device)

        urdf_joint_parameter_names = self.chain.get_joint_parameter_names()
        assert set(self.urdf_joint_ids) == set(urdf_joint_parameter_names), "Joint name mismatch between the user defined urdf joint_ids and the actual joint names in the URDF file. Please check if your config.yaml and URDF file have the same hand type (left/right) and are up to date."
        self.joint_reorder_indices = [urdf_joint_parameter_names.index(name) for name in self.urdf_joint_ids]

        with open(os.path.join(os.path.dirname(__file__), "utils", "retargeter.yaml"), 'r') as file:
            cfg = yaml.safe_load(file)
        self.lr = cfg["lr"]
        self.use_scalar_distance = [False, True, True, True, True] if cfg["use_scalar_distance_palm"] else [False] * 5
        self.joint_regularizers = cfg["joint_regularizers"]
        self.loss_coeffs = torch.tensor(cfg["loss_coeffs"], device=self.device)
        self.orcahand_joint_angles = torch.zeros(len(self.urdf_joint_ids), device=self.device, requires_grad=True)
        self.opt = torch.optim.RMSprop([self.orcahand_joint_angles], lr=self.lr)

        self.root = torch.zeros(1, 3, device=self.device)
        self.regularizer_zeros = torch.zeros(len(self.urdf_joint_ids), device=self.device)
        self.regularizer_weights = torch.zeros(len(self.urdf_joint_ids), device=self.device)
        for joint_id, zero_val, weight in self.joint_regularizers:
            idx = self.joint_ids.index(joint_id)
            self.regularizer_zeros[idx] = zero_val
            self.regularizer_weights[idx] = weight
        for i in range(len(self.joint_ids)):
            if self.regularizer_weights[i] > 0:
                self.regularizer_zeros[i] -= ref_offsets_deg[i]

        self.urdfhand_center, self.urdfhand_rot_matrix, self.optimization_frames = retargeter_utils.get_urdf_model_params(
            self.chain, self.hand_type, self.fingers, self.root)

        self._fingertip_offsets = retargeter_utils.get_fingertip_offset_tensors(self.fingers, self.device)

        # Compute URDF key vector magnitudes at neutral config (for auto-scaling)
        # Use halfway between curled (0) and extended (-ref) to approximate relaxed pose
        neutral_angles = torch.zeros(self.chain.n_joints, device=self.device)
        ref_rad_tensor = torch.tensor(
            retargeter_utils.get_ref_offsets_array(self.joint_ids, self._ref_offsets),
            device=self.device, dtype=torch.float32)
        neutral_angles[self.joint_reorder_indices] = -0.5 * ref_rad_tensor
        urdf_fingertips, urdf_palm = retargeter_utils.extract_orca_fingertips_and_palm(
            self.chain, neutral_angles, self.optimization_frames, self.hand_type, self.fingers, self.root,
            fingertip_offsets=self._fingertip_offsets)
        urdf_keyvectors = retargeter_utils.get_keyvectors(urdf_fingertips, urdf_palm)
        self._urdf_keyvector_mags = 0.9 * np.array([kv.detach().cpu().norm().item() for kv in urdf_keyvectors])

        # Manual calibration (adjustable at runtime via viewer sliders)
        self.manual_scale = 1.0
        self.manual_rotation = np.zeros(3)
        self.manual_translation = np.zeros(3)

        # Auto-scale calibration state
        self.mano_scale = 1.0
        self._calibration_frames = 30
        self._calibration_mags = []
        self._frame_count = 0
        self.fk_points = None
        self.enable_viz = True
        self._blend_frames = 30  # frames to blend from calibration to optimization
        self._blend_count = 0
        self._prev_angles = None

        # Build loss closure with pre-resolved frame names and offsets
        self._loss_fn = _build_loss_fn(
            self.chain, self.joint_reorder_indices, self.optimization_frames,
            self.hand_type, self.fingers, self.root, self._fingertip_offsets,
            self.loss_coeffs, self.use_scalar_distance,
            self.regularizer_weights, self.regularizer_zeros)


    def optimize_orcahand_joint_angles(self, manohand_joint_pos: np.ndarray, opt_steps: int = 2) -> Tuple[np.ndarray, float]:

        manohand_joint_pos = torch.from_numpy(manohand_joint_pos).to(self.device)
        manohand_fingertips, manohand_palm = retargeter_utils.extract_mano_fingertips_and_palm(manohand_joint_pos, self.fingers, self.source)

        # Stack MANO fingertips into (5, 1, 3) tensor for compiled loss function
        mano_tips_stacked = torch.stack([manohand_fingertips[f] for f in self.fingers])
        mano_palm = manohand_palm

        for _ in range(opt_steps):
            loss = self._loss_fn(self.orcahand_joint_angles, mano_tips_stacked, mano_palm)

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            with torch.no_grad():
                self.orcahand_joint_angles.clamp_(self.joint_angle_limits_lower, self.joint_angle_limits_upper)

        return self.orcahand_joint_angles.detach().cpu().numpy()


    def retarget(self, data: np.ndarray, manual_wrist_angle: Union[float, None] = None) -> Dict[str, float]:
        """Retarget MANO data to Orca Hand joint angles."""
        
        if self.source == "avp":
            joints, computed_wrist_angle = retargeter_utils.preprocess_avp_data(data, self.hand_type)
        elif self.source == "mediapipe":
            joints, computed_wrist_angle = retargeter_utils.preprocess_mediapipe_data(data)
        elif self.source == "multicam":
            joints, computed_wrist_angle = retargeter_utils.preprocess_multicam_data(data)
        elif self.source == "manus":
            joints, computed_wrist_angle = retargeter_utils.preprocess_manus_data(data)
        else:
             raise ValueError(f"Unsupported source: {self.source}")

        final_wrist_angle = manual_wrist_angle if manual_wrist_angle is not None else computed_wrist_angle
        # Normalize MANO joint positions to local urdf hand coordinate system
        manohand_joint_pos = retargeter_utils.get_normalized_local_manohand_joint_pos(joints, self.source)

        # Auto-scale calibration: collect MANO key vector magnitudes for first N valid frames
        if len(self._calibration_mags) < self._calibration_frames:
            mano_t = torch.from_numpy(manohand_joint_pos).to(self.device)
            mano_ft, mano_palm = retargeter_utils.extract_mano_fingertips_and_palm(mano_t, self.fingers, self.source)
            mano_kvs = retargeter_utils.get_keyvectors(mano_ft, mano_palm)
            mano_mags = np.array([kv.detach().cpu().norm().item() for kv in mano_kvs])
            self._calibration_mags.append(mano_mags)
            if len(self._calibration_mags) == self._calibration_frames:
                all_mags = np.array(self._calibration_mags)
                median_mano_mags = np.median(all_mags, axis=0)
                ratios = self._urdf_keyvector_mags / np.clip(median_mano_mags, 1e-6, None)
                self.mano_scale = float(np.median(ratios))
                print(f"Auto-scale calibrated: {self.mano_scale:.4f}")
                for i, finger in enumerate(self.fingers):
                    print(f"  {finger}: URDF={self._urdf_keyvector_mags[i]:.4f} MANO={median_mano_mags[i]:.4f} ratio={ratios[i]:.4f}")
                with torch.no_grad():
                    self.orcahand_joint_angles.zero_()
                self.opt = torch.optim.RMSprop([self.orcahand_joint_angles], lr=self.lr)

        manohand_joint_pos = manohand_joint_pos * self.mano_scale
        manohand_joint_pos = manohand_joint_pos @ self.urdfhand_rot_matrix.T + self.urdfhand_center + np.array([0, 0, -0.02])
        manohand_joint_pos = apply_manual_calibration(
            manohand_joint_pos, self.urdfhand_center,
            self.manual_scale, self.manual_rotation, self.manual_translation)

        if len(self._calibration_mags) < self._calibration_frames:
            zero_angles = np.zeros(len(self.urdf_joint_ids))
            final_wrist_angle = np.clip(final_wrist_angle, self.wrist_limit_lower, self.wrist_limit_upper)
            zero_angles[self.wrist_idx] = final_wrist_angle if self.hand_type == "left" else -final_wrist_angle
            self.target_angles = zero_angles
            self._prev_angles = zero_angles.copy()
            self.mano_points = retargeter_utils.rotate_points_around_x(manohand_joint_pos, final_wrist_angle, self.source, self.hand_type)
            return {urdf_joint_id: np.deg2rad(angle) for urdf_joint_id, angle in zip(self.urdf_joint_ids, zero_angles)}

        optimized_angles = self.optimize_orcahand_joint_angles(manohand_joint_pos)

        # Smooth blend from calibration pose to optimized angles over _blend_frames
        if self._blend_count < self._blend_frames:
            alpha = (self._blend_count + 1) / self._blend_frames
            if self._prev_angles is not None:
                optimized_angles = self._prev_angles * (1 - alpha) + optimized_angles * alpha
            self._blend_count += 1
        self._prev_angles = optimized_angles.copy()

        self._frame_count += 1
        if self.verbose and self._frame_count % 60 == 1:
            for finger in self.fingers:
                abd = f"{finger}_abd" if finger != "thumb" else "thumb_abd"
                mcp = f"{finger}_mcp" if finger != "thumb" else "thumb_cmc"
                pip = f"{finger}_pip" if finger != "thumb" else "thumb_mcp"
                a = optimized_angles[self.joint_ids.index(abd)] if abd in self.joint_ids else 0
                m = optimized_angles[self.joint_ids.index(mcp)]
                p = optimized_angles[self.joint_ids.index(pip)]
                print(f"  {finger:6s}: ABD={a:6.1f}° MCP={m:6.1f}° PIP={p:6.1f}°")

        # Wrist angle is inverted for right hand due to URDF inconsistency, should be fixed/standardized in future URDF update
        final_wrist_angle = np.clip(final_wrist_angle, self.wrist_limit_lower, self.wrist_limit_upper)
        optimized_angles[self.wrist_idx] = final_wrist_angle if self.hand_type == "left" else -final_wrist_angle
        self.target_angles = optimized_angles

        if self.enable_viz:
            with torch.no_grad():
                viz_angles = torch.zeros(self.chain.n_joints, device=self.device)
                viz_angles[self.joint_reorder_indices] = torch.tensor(optimized_angles, device=self.device, dtype=torch.float32) / (180.0 / np.pi)
                viz_ft, viz_palm = retargeter_utils.extract_orca_fingertips_and_palm(
                    self.chain, viz_angles, self.optimization_frames, self.hand_type, self.fingers, self.root,
                    fingertip_offsets=self._fingertip_offsets)
                self.fk_points = np.array([viz_ft[f].cpu().numpy().squeeze() for f in self.fingers] + [viz_palm.cpu().numpy().squeeze()])
            self.mano_points = retargeter_utils.rotate_points_around_x(manohand_joint_pos, final_wrist_angle, self.source, self.hand_type)

        return {urdf_joint_id: np.deg2rad(angle) for urdf_joint_id, angle in zip(self.urdf_joint_ids, optimized_angles)}
