"""Train GeoRT IK model without SAPIEN dependency.

Uses the pre-trained FK model checkpoint + human data to train the IK retargeting
model with GeoRT's geometric losses (direction, chamfer, curvature, pinch).

Usage:
    python scripts/geort/train_ik_standalone.py \
        --hand orca_right \
        --human-data human_fred \
        --source manus \
        --tag manus_v3 \
        --epochs 200
"""
import os
import sys
import argparse
import math
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from datetime import datetime

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.join(_SCRIPT_DIR, "..", "..")
_GEORT_ROOT = os.path.join(_PROJECT_ROOT, "third_party", "GeoRT")
sys.path.insert(0, _GEORT_ROOT)

from geort.utils.config_utils import get_config, save_json
from geort.model import FKModel, IKModel
from geort.formatter import HandFormatter
from geort.dataset import RobotKinematicsDataset, MultiPointDataset
from geort.loss import chamfer_distance


def format_loss(value):
    return f"{value:.4e}" if math.fabs(value) < 1e-3 else f"{value:.4f}"


def main():
    parser = argparse.ArgumentParser(description="Train GeoRT IK model (no SAPIEN)")
    parser.add_argument("--hand", type=str, required=True)
    parser.add_argument("--human-data", type=str, required=True,
                        help="Name of human data file in third_party/GeoRT/data/<name>.npy")
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--w-chamfer", type=float, default=80.0)
    parser.add_argument("--w-curvature", type=float, default=0.1)
    parser.add_argument("--w-pinch", type=float, default=1.0)
    parser.add_argument("--source", type=str, default="manus", choices=["manus", "mediapipe"],
                        help="Source type that collected the human data (determines fingertip indices)")
    args = parser.parse_args()

    os.chdir(_GEORT_ROOT)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    config = get_config(args.hand)
    joint_order = config["joint_order"]

    # Parse keypoint info
    keypoint_joints = []
    keypoint_links = []
    finger_names = []
    for info in config["fingertip_link"]:
        keypoint_joints.append([joint_order.index(j) for j in info["joint"]])
        keypoint_links.append(info["link"])
        finger_names.append(info["name"])

    # Source-aware fingertip indices
    FINGERTIP_INDICES = {
        "manus": {"thumb": 24, "index": 4, "middle": 9, "ring": 19, "pinky": 14},
        "mediapipe": {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20},
    }
    tip_map = FINGERTIP_INDICES[args.source]
    human_ids = [tip_map[name] for name in finger_names]
    print(f"Fingers: {finger_names}")
    print(f"Source: {args.source}")
    print(f"Keypoint joints: {keypoint_joints}")
    print(f"Fingertip indices: {human_ids}")

    # Joint limits from URDF
    import xml.etree.ElementTree as ET
    urdf_path = os.path.join(_GEORT_ROOT, config["urdf_path"].lstrip("./"))
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
    normalizer = HandFormatter(joint_lower, joint_upper)

    # Load frozen FK model (3D output per finger)
    fk_model = FKModel(keypoint_joints=keypoint_joints).to(device)
    fk_ckpt = f"checkpoint/fk_model_{config['name']}.pth"
    fk_model.load_state_dict(torch.load(fk_ckpt, map_location=device))
    fk_model.eval()
    for p in fk_model.parameters():
        p.requires_grad = False
    print(f"Loaded FK model from {fk_ckpt}")

    # IK model (3 inputs per finger: fingertip xyz)
    ik_model = IKModel(keypoint_joints=keypoint_joints).to(device)
    ik_optim = optim.AdamW(ik_model.parameters(), lr=1e-4)

    # Robot pointcloud for chamfer loss (from FK dataset)
    fk_dataset = RobotKinematicsDataset(f"data/{config['name']}.npz", keypoint_names=keypoint_links)
    robot_points = fk_dataset.export_robot_pointcloud(keypoint_links)  # [N_fingers, N_samples, 3]
    print(f"Robot pointcloud shape: {robot_points.shape}")

    # Human data
    human_data_path = os.path.join("data", f"{args.human_data}.npy")
    if not os.path.exists(human_data_path):
        human_data_path = os.path.join("data", args.human_data, "human_data.npy")
    human_points_raw = np.load(human_data_path)
    print(f"Human data shape: {human_points_raw.shape}")

    # Rotate human data from GeoRT canonical frame to URDF palm-local frame (90deg around Z)
    canonical_to_palm = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=np.float32)
    human_points_raw = human_points_raw @ canonical_to_palm
    print(f"Applied canonical->palm frame rotation")

    # Extract fingertip positions only (3D per finger)
    human_points = np.array([human_points_raw[:, tid, :3] for tid in human_ids])  # [N_fingers, N_frames, 3]
    for link, hid in zip(keypoint_links, human_ids):
        print(f"  {link} <- tip={hid}: range [{human_points_raw[:, hid, :3].min():.4f}, {human_points_raw[:, hid, :3].max():.4f}]")

    # Skip voxel resampling — sample directly
    n_target = 20000
    n_frames = human_points.shape[1]
    if n_frames >= n_target:
        indices = np.random.choice(n_frames, n_target, replace=False)
    else:
        indices = np.random.choice(n_frames, n_target, replace=True)
    sampled_points = human_points[:, indices, :].astype(np.float32)
    point_dataset = MultiPointDataset(sampled_points)
    point_dataloader = DataLoader(point_dataset, batch_size=2048, shuffle=True)

    # Checkpoint dirs
    timestring = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    save_dir = Path(f"./checkpoint/{config['name']}_{timestring}")
    if args.tag:
        save_dir = Path(f"{save_dir}_{args.tag}")
    last_save_dir = Path(f"./checkpoint/{config['name']}_last")
    save_dir.mkdir(parents=True, exist_ok=True)
    last_save_dir.mkdir(parents=True, exist_ok=True)

    # Save config with joint limits
    export_config = config.copy()
    export_config["joint"] = {
        "lower": joint_lower.tolist(),
        "upper": joint_upper.tolist()
    }
    save_json(export_config, save_dir / "config.json")
    save_json(export_config, last_save_dir / "config.json")

    n_keypoints = len(keypoint_links)

    print(f"\nTraining IK model ({args.epochs} epochs)...")
    print(f"  Chamfer: {args.w_chamfer}, Curvature: {args.w_curvature}, Pinch: {args.w_pinch}")

    for epoch in range(args.epochs):
        for batch_idx, batch in enumerate(point_dataloader):
            point = batch.to(device)  # [B, N_fingers, 3]

            joint = ik_model(point)          # [B, DOF]
            embedded_point = fk_model(joint)  # [B, N_fingers, 3]

            # Pinch loss
            n_finger = point.size(1)
            pinch_loss = 0
            for i in range(n_finger):
                for j in range(i + 1, n_finger):
                    distance = point[:, i, ...] - point[:, j, ...]
                    mask = (torch.norm(distance, dim=-1) < 0.015).float()
                    e_distance = ((embedded_point[:, i, ...] - embedded_point[:, j, ...]) ** 2).sum(dim=-1)
                    pinch_loss += (mask * e_distance).mean() / (mask.sum() + 1e-7) * point.size(0)

            # Curvature loss
            direction = F.normalize(torch.randn_like(point), dim=-1, p=2)
            scale = 0.002
            delta1 = direction * scale
            embedded_point_p = fk_model(ik_model(point + delta1))
            embedded_point_n = fk_model(ik_model(point - delta1))
            curvature_loss = ((embedded_point_p + embedded_point_n - 2 * embedded_point) ** 2).mean()

            # Chamfer loss
            selected_idx = np.random.randint(0, robot_points.shape[1], 2048)
            target = torch.from_numpy(robot_points[:, selected_idx, :]).permute(1, 0, 2).float().to(device)
            chamfer_loss = 0
            for i in range(n_keypoints):
                chamfer_loss += chamfer_distance(embedded_point[:, i, :].unsqueeze(0), target[:, i, :].unsqueeze(0))

            # Direction loss
            direction = F.normalize(torch.randn_like(point), dim=-1, p=2)
            scale = 0.001 + torch.rand(point.size(0)).to(device).unsqueeze(-1).unsqueeze(-1) * 0.01
            point_delta = point + direction * scale
            embedded_point_delta = fk_model(ik_model(point_delta))
            d1 = (point_delta - point).reshape(-1, 3)
            d2 = (embedded_point_delta - embedded_point).reshape(-1, 3)
            direction_loss = -(((F.normalize(d1, dim=-1, p=2, eps=1e-5) * F.normalize(d2, dim=-1, p=2, eps=1e-5)).sum(-1))).mean()

            collision_loss = torch.tensor([0.0]).to(device)

            loss = direction_loss + \
                   chamfer_loss * args.w_chamfer + \
                   curvature_loss * args.w_curvature + \
                   collision_loss * 0.0 + \
                   pinch_loss * args.w_pinch

            ik_optim.zero_grad()
            loss.backward()
            ik_optim.step()

            if batch_idx % 50 == 0:
                print(
                    f"Epoch {epoch} Batch {batch_idx} | "
                    f"Dir: {format_loss(direction_loss.item())} "
                    f"Cham: {format_loss(chamfer_loss.item())} "
                    f"Curv: {format_loss(curvature_loss.item())} "
                    f"Pinch: {format_loss(pinch_loss.item() if isinstance(pinch_loss, torch.Tensor) else pinch_loss)}"
                )

        torch.save(ik_model.state_dict(), save_dir / f"epoch_{epoch}.pth")
        torch.save(ik_model.state_dict(), save_dir / "last.pth")
        torch.save(ik_model.state_dict(), last_save_dir / f"epoch_{epoch}.pth")
        torch.save(ik_model.state_dict(), last_save_dir / "last.pth")

    print(f"\nTraining complete!")
    print(f"Checkpoints: {save_dir}")
    print(f"Last checkpoint: {last_save_dir}")


if __name__ == "__main__":
    main()
