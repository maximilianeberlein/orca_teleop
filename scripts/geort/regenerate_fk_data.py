"""Regenerate FK training data with actual fingertip positions.

The original orca_right.npz was generated using link origins from pytorch_kinematics.
The 'fingertip' link origins are actually at the PIP/DIP joint positions (where the
last revolute joint connects to the fingertip link), NOT the physical fingertip.

This script recomputes fingertip positions as:
    actual_tip = fingertip_link_origin + fingertip_link_rotation @ [0, 0, offset]
where offset is the distal phalanx length measured from the STL mesh extent.

This makes the FK data sensitive to PIP/DIP joint angles, which is critical for
training an IK model that can correctly predict these angles.

Usage:
    python scripts/geort/regenerate_fk_data.py --hand orca_right
"""
import os
import sys
import argparse
import numpy as np
import torch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.join(_SCRIPT_DIR, "..", "..")
_GEORT_ROOT = os.path.join(_PROJECT_ROOT, "third_party", "GeoRT")
sys.path.insert(0, _GEORT_ROOT)

import pytorch_kinematics as pk
from geort.utils.config_utils import get_config


# Distal phalanx length (mm) from STL mesh max Z extent.
# Measured from fingertip link origin (PIP/DIP joint) to physical fingertip.
FINGERTIP_OFFSETS_MM = {
    "thumb": 30.5,
    "index": 43.3,
    "middle": 45.3,
    "ring": 45.3,
    "pinky": 38.3,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hand", type=str, required=True)
    parser.add_argument("--n-samples", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.chdir(_GEORT_ROOT)
    config = get_config(args.hand)
    joint_order = config["joint_order"]

    urdf_path = config["urdf_path"].lstrip("./")
    with open(urdf_path) as f:
        chain = pk.build_chain_from_urdf(f.read())

    urdf_joint_names = chain.get_joint_parameter_names(exclude_fixed=True)
    print(f"Config joints ({len(joint_order)}): {joint_order}")
    print(f"URDF joints ({len(urdf_joint_names)}): {urdf_joint_names}")

    # Build mapping from config joint order to URDF joint order
    config_to_urdf = [urdf_joint_names.index(j) for j in joint_order]

    # Parse joint limits from URDF
    import xml.etree.ElementTree as ET
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    urdf_limits = {}
    for joint in root.findall("joint"):
        name = joint.get("name")
        limit = joint.find("limit")
        if limit is not None and joint.get("type") == "revolute":
            urdf_limits[name] = (float(limit.get("lower")), float(limit.get("upper")))

    joint_lower = np.array([urdf_limits[j][0] for j in joint_order], dtype=np.float32)
    joint_upper = np.array([urdf_limits[j][1] for j in joint_order], dtype=np.float32)

    # Parse fingertip link names and finger names from config
    keypoint_links = []
    finger_names = []
    for info in config["fingertip_link"]:
        keypoint_links.append(info["link"])
        finger_names.append(info["name"])
    print(f"Fingertip links: {keypoint_links}")
    print(f"Finger names: {finger_names}")

    # Fingertip offsets in meters (along local Z of the fingertip link)
    offsets = np.array([FINGERTIP_OFFSETS_MM[n] / 1000.0 for n in finger_names], dtype=np.float32)
    print(f"Fingertip offsets (mm): {[FINGERTIP_OFFSETS_MM[n] for n in finger_names]}")

    # Sample random joint angles
    np.random.seed(args.seed)
    qpos = np.random.uniform(joint_lower, joint_upper, size=(args.n_samples, len(joint_order))).astype(np.float32)
    print(f"\nGenerating {args.n_samples} samples...")

    # Process in batches for memory efficiency
    batch_size = 1000
    keypoint_data = {link: np.zeros((args.n_samples, 3), dtype=np.float32) for link in keypoint_links}
    # Also store PIP joint positions (link origins) for potential PIP loss
    pip_joint_data = {link: np.zeros((args.n_samples, 3), dtype=np.float32) for link in keypoint_links}

    # Get palm position for centering (same as original data generation)
    zero_fk = chain.forward_kinematics(torch.zeros(1, len(urdf_joint_names)))
    palm_pos = zero_fk["right_palm"].get_matrix()[0, :3, 3].numpy()
    palm_rot = zero_fk["right_palm"].get_matrix()[0, :3, :3].numpy()
    print(f"Palm position: {palm_pos}")
    print(f"Palm rotation:\n{palm_rot}")

    for batch_start in range(0, args.n_samples, batch_size):
        batch_end = min(batch_start + batch_size, args.n_samples)
        batch_qpos = qpos[batch_start:batch_end]
        bs = batch_end - batch_start

        # Map config joint order to URDF joint order
        urdf_angles = torch.zeros(bs, len(urdf_joint_names))
        for ci, ui in enumerate(config_to_urdf):
            urdf_angles[:, ui] = torch.from_numpy(batch_qpos[:, ci])

        fk_result = chain.forward_kinematics(urdf_angles)

        for f_idx, (link_name, f_name) in enumerate(zip(keypoint_links, finger_names)):
            mat = fk_result[link_name].get_matrix()  # [bs, 4, 4]
            origins = mat[:, :3, 3].numpy()  # [bs, 3] — PIP/DIP joint position
            local_z = mat[:, :3, 2].numpy()  # [bs, 3] — local Z axis
            actual_tips = origins + local_z * offsets[f_idx]  # [bs, 3]

            # Transform to palm-local frame (same as original data)
            pip_palm = (origins - palm_pos) @ palm_rot  # [bs, 3]
            tip_palm = (actual_tips - palm_pos) @ palm_rot  # [bs, 3]

            keypoint_data[link_name][batch_start:batch_end] = tip_palm
            pip_joint_data[link_name][batch_start:batch_end] = pip_palm

        if (batch_start // batch_size) % 10 == 0:
            print(f"  Processed {batch_end}/{args.n_samples}")

    print(f"Processed all {args.n_samples} samples")

    # Verify: check ranges
    print("\n=== Actual fingertip positions (palm-local frame) ===")
    for link_name, f_name in zip(keypoint_links, finger_names):
        pos = keypoint_data[link_name]
        print(f"  {f_name}: x=[{pos[:,0].min():.4f},{pos[:,0].max():.4f}] "
              f"y=[{pos[:,1].min():.4f},{pos[:,1].max():.4f}] "
              f"z=[{pos[:,2].min():.4f},{pos[:,2].max():.4f}]")

    print("\n=== PIP joint positions (palm-local frame) ===")
    for link_name, f_name in zip(keypoint_links, finger_names):
        pos = pip_joint_data[link_name]
        print(f"  {f_name}: x=[{pos[:,0].min():.4f},{pos[:,0].max():.4f}] "
              f"y=[{pos[:,1].min():.4f},{pos[:,1].max():.4f}] "
              f"z=[{pos[:,2].min():.4f},{pos[:,2].max():.4f}]")

    # Save — backup old file first
    out_path = f"data/{config['name']}.npz"
    backup_path = f"data/{config['name']}_old_pip_joint_positions.npz"
    if os.path.exists(out_path) and not os.path.exists(backup_path):
        os.rename(out_path, backup_path)
        print(f"\nBacked up old data to {backup_path}")

    # Save in same format as original: qpos + keypoint dict
    # Also add pip_joint positions for PIP loss during IK training
    np.savez(out_path,
             qpos=qpos,
             keypoint=keypoint_data,
             pip_joint=pip_joint_data)
    print(f"Saved to {out_path}")
    print(f"  qpos: {qpos.shape}")
    print(f"  keypoint: {len(keypoint_data)} links x {args.n_samples} samples")
    print(f"  pip_joint: {len(pip_joint_data)} links x {args.n_samples} samples")


if __name__ == "__main__":
    main()
