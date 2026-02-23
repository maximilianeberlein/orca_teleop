import os
import functools
from typing import Dict, Union
import numpy as np
import torch
import yaml
from orca_core import OrcaHand
from .utils import retargeter_utils
from .utils.manual_calibration import apply_manual_calibration

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxls
import pyroki as pk
import yourdfpy
from .utils.retargeter_utils import FINGERTIP_OFFSETS


def _quat_rotate_jax(q_wxyz, v):
    w, x, y, z = q_wxyz[0], q_wxyz[1], q_wxyz[2], q_wxyz[3]
    t = 2.0 * jnp.cross(jnp.array([x, y, z]), v)
    return v + w * t + jnp.cross(jnp.array([x, y, z]), t)


def _quat_rotate_np(q_wxyz, v):
    w, x, y, z = q_wxyz[0], q_wxyz[1], q_wxyz[2], q_wxyz[3]
    t = 2.0 * np.cross(np.array([x, y, z]), v)
    return v + w * t + np.cross(np.array([x, y, z]), t)


def _build_solve_fn(max_iterations, trust_region_lambda_initial, linear_solver):

    @jdc.jit
    def _solve_jax(
        robot: pk.Robot,
        target_keyvectors: jax.Array,
        prev_cfg: jax.Array,
        use_prev_cfg: jax.Array,
        loss_coeffs: jax.Array,
        fingertip_link_indices: jax.Array,
        thumb_base_link_idx: jax.Array,
        pinky_base_link_idx: jax.Array,
        reg_zeros: jax.Array,
        reg_weights: jax.Array,
        smoothness_weight: jax.Array,
        wrist_pyroki_idx: jax.Array,
        coupling_mcp_indices: jax.Array,
        coupling_pip_indices: jax.Array,
        coupling_ratio: jax.Array,
        coupling_weight: jax.Array,
        fingertip_offsets: jax.Array,
    ) -> jax.Array:
        joint_var = robot.joint_var_cls(0)
        variables = [joint_var]

        @jaxls.Cost.factory
        def keyvector_cost(
            vals: jaxls.VarValues,
            var_cfg: jaxls.Var[jax.Array],
            target_kvs: jax.Array,
        ) -> jax.Array:
            cfg = vals[var_cfg]
            fk = robot.forward_kinematics(cfg=cfg)  # (n_links, 7)

            link_quats = fk[fingertip_link_indices, 0:4]  # (5, 4) wxyz
            link_pos = fk[fingertip_link_indices, 4:7]    # (5, 3)
            rotated_offsets = jax.vmap(_quat_rotate_jax)(link_quats, fingertip_offsets)
            fingertip_pos = link_pos + rotated_offsets     # (5, 3)

            thumb_base_pos = fk[thumb_base_link_idx, 4:7]  # (3,)
            pinky_base_pos = fk[pinky_base_link_idx, 4:7]  # (3,)
            palm_pos = (thumb_base_pos + pinky_base_pos) / 2.0 - jnp.array([0.0, 0.0, 0.015])

            urdf_kvs = fingertip_pos - palm_pos[None, :]  # (5, 3)

            weights = jnp.sqrt(loss_coeffs)  # (5,)
            residual = (weights[:, None] * (target_kvs - urdf_kvs)).flatten()  # (15,)
            return residual

        @jaxls.Cost.factory
        def wrist_zero_cost(
            vals: jaxls.VarValues,
            var_cfg: jaxls.Var[jax.Array],
        ) -> jax.Array:
            cfg = vals[var_cfg]
            return jnp.array([100.0 * cfg[wrist_pyroki_idx]])

        @jaxls.Cost.factory
        def regularization_cost(
            vals: jaxls.VarValues,
            var_cfg: jaxls.Var[jax.Array],
        ) -> jax.Array:
            cfg = vals[var_cfg]
            cfg_deg = cfg * (180.0 / jnp.pi)
            residual = jnp.sqrt(reg_weights) * (cfg_deg - reg_zeros)
            return residual

        @jaxls.Cost.factory
        def smoothness_cost(
            vals: jaxls.VarValues,
            var_cfg: jaxls.Var[jax.Array],
            prev: jax.Array,
            weight: jax.Array,
        ) -> jax.Array:
            cfg = vals[var_cfg]
            return jnp.sqrt(weight) * (cfg - prev)

        @jaxls.Cost.factory
        def pip_coupling_cost(
            vals: jaxls.VarValues,
            var_cfg: jaxls.Var[jax.Array],
        ) -> jax.Array:
            cfg = vals[var_cfg]
            mcp_angles = cfg[coupling_mcp_indices]
            pip_angles = cfg[coupling_pip_indices]
            return jnp.sqrt(coupling_weight) * (pip_angles - coupling_ratio * mcp_angles)

        costs = [
            keyvector_cost(joint_var, target_keyvectors),
            regularization_cost(joint_var),
            pk.costs.limit_constraint(robot, joint_var),
            smoothness_cost(joint_var, prev_cfg, smoothness_weight * use_prev_cfg),
            wrist_zero_cost(joint_var),
            pip_coupling_cost(joint_var),
        ]

        initial_vals = jaxls.VarValues.make([joint_var.with_value(
            jnp.where(use_prev_cfg, prev_cfg, jnp.zeros_like(prev_cfg))
        )])

        sol = (
            jaxls.LeastSquaresProblem(costs=costs, variables=variables)
            .analyze()
            .solve(
                initial_vals=initial_vals,
                verbose=False,
                linear_solver=linear_solver,
                trust_region=jaxls.TrustRegionConfig(lambda_initial=trust_region_lambda_initial),
                termination=jaxls.TerminationConfig(max_iterations=max_iterations),
            )
        )
        return sol[joint_var]

    return _solve_jax


class PyRoKIRetargeter:

    def __init__(self, model_path: Union[OrcaHand, str] = None, urdf_path: Union[str, None] = None, source: str = "none") -> None:

        self.source = source
        self.target_angles = None
        self.mano_points = None
        self.fingers = ["thumb", "index", "middle", "ring", "pinky"]

        hand = OrcaHand(model_path)
        if hand.type not in ["left", "right"]:
            raise ValueError("hand.type must be 'left' or 'right'. Update config.yaml with type field.")
        self.hand_type = hand.type
        self.joint_ids = hand.joint_ids
        self.urdf_joint_ids = [f"{hand.type}_{joint_id}" for joint_id in self.joint_ids]

        lower_limits, upper_limits = map(list, zip(*hand.joint_roms_dict.values()))
        self.wrist_limit_lower = lower_limits[16]
        self.wrist_limit_upper = upper_limits[16]
        lower_limits[16] = upper_limits[16] = 0.0

        if not os.path.exists(urdf_path):
            raise ValueError(f"URDF file not found at {urdf_path}")
        urdf_dir = os.path.dirname(os.path.abspath(urdf_path))
        urdf = yourdfpy.URDF.load(
            urdf_path, load_meshes=False,
            filename_handler=functools.partial(yourdfpy.filename_handler_magic, dir=urdf_dir),
        )
        self.robot = pk.Robot.from_urdf(urdf)

        pyroki_joint_names = list(self.robot.joints.actuated_names)
        assert set(self.urdf_joint_ids) == set(pyroki_joint_names), \
            "Joint name mismatch between ORCA config and URDF file."
        # _orca_to_pyroki[i] = PyRoKI index for ORCA's i-th joint
        self._orca_to_pyroki = np.array([pyroki_joint_names.index(name) for name in self.urdf_joint_ids])
        # _pyroki_to_orca[j] = ORCA index for PyRoKI's j-th joint
        self._pyroki_to_orca = np.array([self.urdf_joint_ids.index(name) for name in pyroki_joint_names])
        self._wrist_pyroki_idx = jnp.array(pyroki_joint_names.index(f"{self.hand_type}_wrist"))

        link_names = list(self.robot.links.names)
        self._fingertip_link_indices = jnp.array([
            link_names.index(retargeter_utils.get_fingertip_urdf_name(self.hand_type, f))
            for f in self.fingers
        ])
        self._thumb_base_link_idx = jnp.array(
            link_names.index(retargeter_utils.get_finger_base_urdf_name(self.hand_type, "thumb")))
        self._pinky_base_link_idx = jnp.array(
            link_names.index(retargeter_utils.get_finger_base_urdf_name(self.hand_type, "pinky")))
        self._palm_link_idx = link_names.index(f"{self.hand_type}_palm")

        with open(os.path.join(os.path.dirname(__file__), "utils", "pyroki_retargeter.yaml"), 'r') as file:
            cfg = yaml.safe_load(file)
        self._loss_coeffs = jnp.array(cfg["loss_coeffs"])
        self._smoothness_weight = jnp.array(cfg["smoothness_weight"])

        ref_offsets_deg = np.rad2deg(retargeter_utils.get_ref_offsets_array(self.joint_ids))
        self._lower_limits_deg = np.array(lower_limits) - ref_offsets_deg
        self._upper_limits_deg = np.array(upper_limits) - ref_offsets_deg

        coupling_pairs = [
            ("index_mcp", "index_pip"),
            ("middle_mcp", "middle_pip"),
            ("ring_mcp", "ring_pip"),
            ("pinky_mcp", "pinky_pip"),
            ("thumb_pip", "thumb_dip"),
        ]
        self._coupling_mcp_indices = jnp.array([
            pyroki_joint_names.index(f"{self.hand_type}_{mcp}") for mcp, _ in coupling_pairs
        ])
        self._coupling_pip_indices = jnp.array([
            pyroki_joint_names.index(f"{self.hand_type}_{pip}") for _, pip in coupling_pairs
        ])
        self._coupling_ratio = jnp.array(cfg["pip_coupling_ratio"])
        self._coupling_weight = jnp.array(cfg["pip_coupling_weight"])
        self._coupling_ratio_float = float(cfg["pip_coupling_ratio"])
        self._coupling_orca_pairs = [
            (self.joint_ids.index(mcp), self.joint_ids.index(pip))
            for mcp, pip in coupling_pairs
        ]

        self._solve_jax = _build_solve_fn(
            max_iterations=cfg["max_iterations"],
            trust_region_lambda_initial=cfg["trust_region_lambda_initial"],
            linear_solver=cfg["linear_solver"],
        )

        regularizer_zeros = np.zeros(len(self.urdf_joint_ids))
        regularizer_weights = np.zeros(len(self.urdf_joint_ids))
        for joint_id, zero_val, weight in cfg["joint_regularizers"]:
            idx = self.joint_ids.index(joint_id)
            regularizer_zeros[idx] = zero_val
            regularizer_weights[idx] = weight
        for i in range(len(self.joint_ids)):
            if regularizer_weights[i] > 0:
                regularizer_zeros[i] -= ref_offsets_deg[i]
        # Reorder from ORCA order to PyRoKI joint order:
        # reg_pyroki[j] = reg_orca[_pyroki_to_orca[j]] (value for the joint at PyRoKI position j)
        self._reg_zeros_pyroki = jnp.array(regularizer_zeros[self._pyroki_to_orca])
        self._reg_weights_pyroki = jnp.array(regularizer_weights[self._pyroki_to_orca])

        self._fingertip_offsets_jax = jnp.array([FINGERTIP_OFFSETS[f] for f in self.fingers])  # (5, 3)

        self._compute_urdf_reference_params()

        # Manual calibration (adjustable at runtime via viewer sliders)
        self.manual_scale = 1.0
        self.manual_rotation = np.zeros(3)
        self.manual_translation = np.zeros(3)

        self.mano_scale = 1.0
        self._calibration_frames = 30
        self._calibration_mags = []

        self._prev_cfg = None
        self._frame_count = 0
        n_joints = self.robot.joints.num_actuated_joints
        self._zero_cfg = jnp.zeros(n_joints)

        print("PyRoKI: Compiling JIT solver (this may take a few seconds)...")
        dummy_target = jnp.zeros((5, 3))
        self._solve_jax(
            self.robot, dummy_target,
            self._zero_cfg, jnp.array(False),
            self._loss_coeffs, self._fingertip_link_indices,
            self._thumb_base_link_idx, self._pinky_base_link_idx,
            self._reg_zeros_pyroki, self._reg_weights_pyroki,
            self._smoothness_weight,
            self._wrist_pyroki_idx,
            self._coupling_mcp_indices, self._coupling_pip_indices,
            self._coupling_ratio, self._coupling_weight,
            self._fingertip_offsets_jax,
        )
        print("PyRoKI: JIT compilation complete.")

    def _compute_urdf_reference_params(self):
        extended_orca_rad = -np.array(
            retargeter_utils.get_ref_offsets_array(self.joint_ids), dtype=np.float32)
        extended_cfg = jnp.array(extended_orca_rad[self._pyroki_to_orca])
        fk = self.robot.forward_kinematics(cfg=extended_cfg)  # (n_links, 7)

        def _get_pos(link_idx):
            return np.array(fk[link_idx, 4:7])

        finger_base_positions = {}
        for finger in self.fingers:
            link_name = retargeter_utils.get_finger_base_urdf_name(self.hand_type, finger)
            link_idx = list(self.robot.links.names).index(link_name)
            finger_base_positions[f"{finger}_base"] = _get_pos(link_idx)

        wrist_pos = _get_pos(self._palm_link_idx)
        self.urdfhand_center, self.urdfhand_rot_matrix = retargeter_utils.get_hand_center_and_rotation(
            **finger_base_positions, wrist=wrist_pos)

        fingertip_positions = {}
        for i, f in enumerate(self.fingers):
            link_idx = int(self._fingertip_link_indices[i])
            link_pos = np.array(fk[link_idx, 4:7])
            link_quat = np.array(fk[link_idx, 0:4])  # wxyz
            offset = np.array(FINGERTIP_OFFSETS[f])
            fingertip_positions[f] = link_pos + _quat_rotate_np(link_quat, offset)

        thumb_base = _get_pos(int(self._thumb_base_link_idx))
        pinky_base = _get_pos(int(self._pinky_base_link_idx))
        palm = (thumb_base + pinky_base) / 2.0 - np.array([0, 0, 0.015])

        urdf_keyvectors = [fingertip_positions[f] - palm for f in self.fingers]
        self._urdf_keyvector_mags = np.array([np.linalg.norm(kv) for kv in urdf_keyvectors])

    _PALM_OFFSET = torch.tensor([0, 0, 0.015], dtype=torch.float32)

    def _solve(self, manohand_joint_pos: np.ndarray) -> np.ndarray:
        mano_t = torch.from_numpy(manohand_joint_pos).to("cpu")
        mano_fingertips, mano_palm = retargeter_utils.extract_mano_fingertips_and_palm(mano_t, self.fingers, self.source)
        mano_palm = mano_palm - self._PALM_OFFSET
        keyvectors_mano = retargeter_utils.get_keyvectors(mano_fingertips, mano_palm)

        target_kvs = jnp.array(np.array([kv.detach().numpy().squeeze() for kv in keyvectors_mano]))

        self._frame_count += 1
        if self._frame_count % 60 == 1:
            target_mags = np.linalg.norm(np.array(target_kvs), axis=1) * 1000
            print(f"[PyRoKI diag] target_kv_mags(mm): {' '.join(f'{m:.1f}' for m in target_mags)}  "
                  f"urdf_ref_mags(mm): {' '.join(f'{m:.1f}' for m in self._urdf_keyvector_mags * 1000)}")

        prev_cfg = jnp.array(self._prev_cfg) if self._prev_cfg is not None else self._zero_cfg
        use_prev = jnp.array(self._prev_cfg is not None)

        optimized_cfg = self._solve_jax(
            self.robot, target_kvs,
            prev_cfg, use_prev,
            self._loss_coeffs, self._fingertip_link_indices,
            self._thumb_base_link_idx, self._pinky_base_link_idx,
            self._reg_zeros_pyroki, self._reg_weights_pyroki,
            self._smoothness_weight,
            self._wrist_pyroki_idx,
            self._coupling_mcp_indices, self._coupling_pip_indices,
            self._coupling_ratio, self._coupling_weight,
            self._fingertip_offsets_jax,
        )

        # Reorder from PyRoKI joint order to ORCA joint order:
        # orca[i] = pyroki[_orca_to_pyroki[i]] (pick the PyRoKI value for ORCA's i-th joint)
        orca_angles_rad = np.array(optimized_cfg)[self._orca_to_pyroki]
        orca_angles_deg = np.rad2deg(orca_angles_rad)

        # Hard PIP-MCP coupling: PIP has zero FK gradient for keyvectors,
        # so override with ratio * MCP to guarantee correct PIP tracking
        for mcp_idx, pip_idx in self._coupling_orca_pairs:
            orca_angles_deg[pip_idx] = self._coupling_ratio_float * orca_angles_deg[mcp_idx]

        orca_angles_deg = np.clip(orca_angles_deg, self._lower_limits_deg, self._upper_limits_deg)

        if self._frame_count % 60 == 1:
            for finger in self.fingers:
                abd_name = f"{finger}_abd" if finger != "thumb" else "thumb_abd"
                mcp_name = f"{finger}_mcp" if finger != "thumb" else "thumb_mcp"
                pip_name = f"{finger}_pip" if finger != "thumb" else "thumb_pip"
                abd_val = orca_angles_deg[self.joint_ids.index(abd_name)] if abd_name in self.joint_ids else 0
                mcp_val = orca_angles_deg[self.joint_ids.index(mcp_name)]
                pip_val = orca_angles_deg[self.joint_ids.index(pip_name)]
                print(f"  {finger:6s}: ABD={abd_val:6.1f}° MCP={mcp_val:6.1f}° PIP={pip_val:6.1f}°")

            fk = self.robot.forward_kinematics(cfg=jnp.array(optimized_cfg))
            for i, finger in enumerate(self.fingers):
                link_idx = int(self._fingertip_link_indices[i])
                link_pos = np.array(fk[link_idx, 4:7])
                link_quat = np.array(fk[link_idx, 0:4])
                offset = np.array(FINGERTIP_OFFSETS[finger])
                tip_pos = link_pos + _quat_rotate_np(link_quat, offset)
                tb = np.array(fk[int(self._thumb_base_link_idx), 4:7])
                pb = np.array(fk[int(self._pinky_base_link_idx), 4:7])
                palm = (tb + pb) / 2.0 - np.array([0, 0, 0.015])
                urdf_kv = tip_pos - palm
                tgt_kv = np.array(target_kvs[i])
                print(f"  {finger:6s} kv: target=[{tgt_kv[0]:+.4f},{tgt_kv[1]:+.4f},{tgt_kv[2]:+.4f}]  "
                      f"urdf=[{urdf_kv[0]:+.4f},{urdf_kv[1]:+.4f},{urdf_kv[2]:+.4f}]")

        # Update warm-start with the coupled values (in PyRoKI order)
        self._prev_cfg = np.deg2rad(orca_angles_deg)[self._pyroki_to_orca]
        self._prev_cfg[int(self._wrist_pyroki_idx)] = 0.0

        return orca_angles_deg

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
        manohand_joint_pos = retargeter_utils.get_normalized_local_manohand_joint_pos(joints, self.source)

        if len(self._calibration_mags) < self._calibration_frames:
            mano_t = torch.from_numpy(manohand_joint_pos).to("cpu")
            mano_ft, mano_palm = retargeter_utils.extract_mano_fingertips_and_palm(mano_t, self.fingers, self.source)
            mano_palm = mano_palm - self._PALM_OFFSET
            mano_kvs = retargeter_utils.get_keyvectors(mano_ft, mano_palm)
            mano_mags = np.array([kv.detach().cpu().norm().item() for kv in mano_kvs])
            self._calibration_mags.append(mano_mags)
            if len(self._calibration_mags) == self._calibration_frames:
                all_mags = np.array(self._calibration_mags)
                median_mano_mags = np.median(all_mags, axis=0)
                ratios = self._urdf_keyvector_mags / np.clip(median_mano_mags, 1e-6, None)
                self.mano_scale = float(np.median(ratios))
                print(f"PyRoKI auto-scale calibrated: {self.mano_scale:.4f}")
                for i, finger in enumerate(self.fingers):
                    print(f"  {finger}: URDF={self._urdf_keyvector_mags[i]:.4f} MANO={median_mano_mags[i]:.4f} ratio={ratios[i]:.4f}")
                self._prev_cfg = None

        manohand_joint_pos = manohand_joint_pos * self.mano_scale
        manohand_joint_pos = manohand_joint_pos @ self.urdfhand_rot_matrix.T + self.urdfhand_center + np.array([0, 0, -0.02])
        manohand_joint_pos = apply_manual_calibration(
            manohand_joint_pos, self.urdfhand_center,
            self.manual_scale, self.manual_rotation, self.manual_translation)

        if len(self._calibration_mags) < self._calibration_frames:
            zero_angles = np.zeros(len(self.urdf_joint_ids))
            final_wrist_angle = np.clip(final_wrist_angle, self.wrist_limit_lower, self.wrist_limit_upper)
            zero_angles[-1] = final_wrist_angle if self.hand_type == "left" else -final_wrist_angle
            self.target_angles = zero_angles
            self.mano_points = retargeter_utils.rotate_points_around_y(manohand_joint_pos, final_wrist_angle, self.source, self.hand_type)
            return {urdf_joint_id: np.deg2rad(angle) for urdf_joint_id, angle in zip(self.urdf_joint_ids, zero_angles)}

        optimized_angles = self._solve(manohand_joint_pos)

        final_wrist_angle = np.clip(final_wrist_angle, self.wrist_limit_lower, self.wrist_limit_upper)
        optimized_angles[-1] = final_wrist_angle if self.hand_type == "left" else -final_wrist_angle
        self.target_angles = optimized_angles

        self.mano_points = retargeter_utils.rotate_points_around_y(manohand_joint_pos, final_wrist_angle, self.source, self.hand_type)

        return {urdf_joint_id: np.deg2rad(angle) for urdf_joint_id, angle in zip(self.urdf_joint_ids, optimized_angles)}
