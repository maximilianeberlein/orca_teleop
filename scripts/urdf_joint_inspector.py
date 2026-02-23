#!/usr/bin/env python3
"""Manual URDF joint inspector — sliders for every joint, no retargeting."""

import argparse
import time
import numpy as np
from orca_core import OrcaHand
from orca_teleop.viewer.urdf_viewer import URDFViewer
from orca_teleop.orca_retargeter.utils.retargeter_utils import JOINT_REF_OFFSETS_RAD


def main():
    parser = argparse.ArgumentParser(description="Manually inspect URDF joints with sliders")
    parser.add_argument("model_path", help="Path to OrcaHand model directory")
    parser.add_argument("urdf_path", help="Path to URDF file")
    parser.add_argument("--port", type=int, default=8080, help="Viser server port")
    args = parser.parse_args()

    hand = OrcaHand(args.model_path)
    joint_ids = hand.joint_ids
    urdf_joint_ids = [f"{hand.type}_{jid}" for jid in joint_ids]
    roms = hand.joint_roms_dict

    viewer = URDFViewer(args.urdf_path, port=args.port)
    server = viewer._server

    sliders = {}
    finger_groups = {
        "Thumb": ["thumb_mcp", "thumb_abd", "thumb_pip", "thumb_dip"],
        "Index": ["index_abd", "index_mcp", "index_pip"],
        "Middle": ["middle_abd", "middle_mcp", "middle_pip"],
        "Ring": ["ring_abd", "ring_mcp", "ring_pip"],
        "Pinky": ["pinky_abd", "pinky_mcp", "pinky_pip"],
        "Wrist": ["wrist"],
    }

    for group_name, joints in finger_groups.items():
        with server.gui.add_folder(group_name):
            for jid in joints:
                if jid not in roms:
                    continue
                lo, hi = roms[jid]
                slider = server.gui.add_slider(
                    jid, lo, hi, step=0.5, initial_value=0.0,
                )
                sliders[jid] = slider

    # Preset buttons
    with server.gui.add_folder("Presets"):
        zero_btn = server.gui.add_button("All Zero")
        fist_btn = server.gui.add_button("Fist")
        pinch_btn = server.gui.add_button("Pinch")

    @zero_btn.on_click
    def _(_):
        for s in sliders.values():
            s.value = 0.0

    @fist_btn.on_click
    def _(_):
        for jid, s in sliders.items():
            if "mcp" in jid and "thumb" not in jid:
                s.value = 90.0
            elif "pip" in jid and "thumb" not in jid:
                s.value = 60.0
            elif jid == "thumb_pip":
                s.value = 60.0
            elif jid == "thumb_dip":
                s.value = -60.0
            elif jid == "thumb_mcp":
                s.value = -40.0
            elif "abd" in jid:
                s.value = 0.0
            elif jid == "wrist":
                s.value = 0.0

    @pinch_btn.on_click
    def _(_):
        for s in sliders.values():
            s.value = 0.0
        if "thumb_pip" in sliders:
            sliders["thumb_pip"].value = 40.0
        if "thumb_dip" in sliders:
            sliders["thumb_dip"].value = -40.0
        if "index_mcp" in sliders:
            sliders["index_mcp"].value = 50.0
        if "index_pip" in sliders:
            sliders["index_pip"].value = 33.0

    print("URDF Joint Inspector running. Open the browser to adjust sliders.")
    print("Press Ctrl+C or click Stop to exit.")

    while not viewer.stopped:
        angles = {}
        for jid, slider in sliders.items():
            urdf_name = f"{hand.type}_{jid}"
            ref_deg = np.rad2deg(JOINT_REF_OFFSETS_RAD.get(jid, 0.0))
            angles[urdf_name] = np.deg2rad(slider.value - ref_deg)
        viewer.update(angles)
        time.sleep(0.03)

    viewer.close()


if __name__ == "__main__":
    main()
