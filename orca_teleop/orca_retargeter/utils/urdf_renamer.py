"""Auto-detect and rename Onshape-exported URDF files to semantic joint/link names.

Onshape URDFs use auto-generated names like ``P-AP_f5e42b61`` for links and
``P-AP_f5e42b61_to_R-Carpals_8d1f1041`` for joints. The retargeter expects
semantic names like ``right_pinky_abd``. This module detects Onshape naming
and renames in-place (with backup) so the teleop pipeline works transparently.
"""

import json
import math
import os
import re
import shutil
import xml.etree.ElementTree as ET


def _strip_hash(name: str) -> str:
    """Remove trailing ``_[0-9a-f]{8}`` Onshape hex suffix from a name."""
    return re.sub(r"_[0-9a-f]{8}$", "", name)


# ---------------------------------------------------------------------------
# Link prefix → semantic role mapping
# ---------------------------------------------------------------------------
# Sorted longest-first so ``I-AP-R`` matches before ``I-AP``, etc.
# The key is matched with str.startswith against the hash-stripped link name.
_LINK_PREFIX_MAP = [
    ("R-T-AP",              "thumb_abd_link"),
    ("R-Carpals",           "palm"),
    ("P-AP",                "pinky_mp"),
    ("P-PP",                "pinky_pp"),
    ("P-FingerTip",         "pinky_fingertip"),
    ("I-AP",                "index_mp"),
    ("I-PP",                "index_pp"),
    ("I-FingerTip",         "index_fingertip"),
    ("T-TP",                "thumb_mp"),
    ("T-PP",                "thumb_pp"),
    ("T-DP",                "thumb_fingertip"),
    ("TopTower",            "wrist_link"),
    ("ForeArm",             "forearm"),
    # M-AP / M-PP / M-FingerTip are handled separately (middle/ring disambiguation)
]


def _match_link_prefix(stripped: str):
    """Return the semantic role for a stripped link name, or None."""
    for prefix, role in _LINK_PREFIX_MAP:
        if stripped.startswith(prefix):
            return role
    return None


# ---------------------------------------------------------------------------
# Joint map: (child_role, parent_role) → bare joint name
# ---------------------------------------------------------------------------
_JOINT_MAP = {
    ("palm", "wrist_link"):                 "wrist",
    ("pinky_mp", "palm"):                   "pinky_abd",
    ("pinky_pp", "pinky_mp"):               "pinky_mcp",
    ("pinky_fingertip", "pinky_pp"):        "pinky_pip",
    ("middle_mp", "palm"):                  "middle_abd",
    ("middle_pp", "middle_mp"):             "middle_mcp",
    ("middle_fingertip", "middle_pp"):      "middle_pip",
    ("ring_mp", "palm"):                    "ring_abd",
    ("ring_pp", "ring_mp"):                 "ring_mcp",
    ("ring_fingertip", "ring_pp"):          "ring_pip",
    ("index_mp", "palm"):                   "index_abd",
    ("index_pp", "index_mp"):               "index_mcp",
    ("index_fingertip", "index_pp"):        "index_pip",
    ("thumb_mp", "palm"):                   "thumb_cmc",
    ("thumb_abd_link", "thumb_mp"):         "thumb_abd",
    ("thumb_pp", "thumb_abd_link"):         "thumb_mcp",
    ("thumb_fingertip", "thumb_pp"):        "thumb_dip",
    ("wrist_link", "forearm"):              "wrist_link_fixed",
}


# ---------------------------------------------------------------------------
# Middle / ring disambiguation
# ---------------------------------------------------------------------------

def _disambiguate_middle_ring(tree: ET.ElementTree):
    """Identify which of the two M-AP chains is middle vs ring.

    Strategy: find the two M-AP links' joint origin x-positions relative to
    the palm (R-Carpals).  More negative x = closer to index = middle finger.
    More positive x = closer to pinky = ring finger.

    Returns a dict mapping original link name → semantic role for all M-* links.
    """
    root = tree.getroot()

    # Collect all M-AP links and their joint origin x-positions
    m_ap_links = []  # [(link_name, x_position)]
    for joint_el in root.iter("joint"):
        child_el = joint_el.find("child")
        if child_el is None:
            continue
        child_name = child_el.get("link")
        stripped = _strip_hash(child_name)
        if stripped.startswith("M-AP"):
            origin_el = joint_el.find("origin")
            x = 0.0
            if origin_el is not None:
                xyz = origin_el.get("xyz", "0 0 0").split()
                x = float(xyz[0])
            m_ap_links.append((child_name, x))

    if len(m_ap_links) != 2:
        raise ValueError(
            f"Expected exactly 2 M-AP links for middle/ring disambiguation, "
            f"found {len(m_ap_links)}: {m_ap_links}"
        )

    # Sort by x-position: more negative = middle, more positive = ring
    m_ap_links.sort(key=lambda pair: pair[1])
    middle_ap_name, ring_ap_name = m_ap_links[0][0], m_ap_links[1][0]

    # Trace each chain: M-AP → M-PP → M-FingerTip
    def _trace_chain(ap_name, finger):
        """Follow child joints from an M-AP link to build the rename map."""
        mapping = {ap_name: f"{finger}_mp"}
        current = ap_name
        role_sequence = [f"{finger}_pp", f"{finger}_fingertip"]
        role_idx = 0
        for joint_el in root.iter("joint"):
            parent_el = joint_el.find("parent")
            if parent_el is None:
                continue
            if parent_el.get("link") == current and role_idx < len(role_sequence):
                child_name = joint_el.find("child").get("link")
                mapping[child_name] = role_sequence[role_idx]
                current = child_name
                role_idx += 1
        return mapping

    result = {}
    result.update(_trace_chain(middle_ap_name, "middle"))
    result.update(_trace_chain(ring_ap_name, "ring"))
    return result


# ---------------------------------------------------------------------------
# MuJoCo ref offset extraction
# ---------------------------------------------------------------------------

def _find_mujoco_xml(urdf_path: str):
    """Search ancestor directories of *urdf_path* for a MuJoCo XML file.

    Returns the path to the first ``*.xml`` file whose root tag is ``<mujoco>``,
    or ``None`` if nothing is found.  Searches up to 3 levels above the URDF.
    """
    search_dir = os.path.dirname(os.path.abspath(urdf_path))
    for _ in range(4):  # current dir + 3 parents
        for fname in os.listdir(search_dir):
            if not fname.endswith(".xml"):
                continue
            candidate = os.path.join(search_dir, fname)
            try:
                tree = ET.parse(candidate)
                if tree.getroot().tag == "mujoco":
                    return candidate
            except ET.ParseError:
                continue
        parent = os.path.dirname(search_dir)
        if parent == search_dir:
            break
        search_dir = parent
    return None


def _normalize_angle(rad: float) -> float:
    """Normalize an angle in radians to [-pi, pi]."""
    rad = rad % (2 * math.pi)
    if rad > math.pi:
        rad -= 2 * math.pi
    return rad


def _extract_ref_offsets(mujoco_path: str, link_to_bare_joint: dict) -> dict:
    """Parse MuJoCo XML and extract ``ref`` attributes mapped to bare joint names.

    *link_to_bare_joint* maps original (pre-rename) link names to bare joint
    names (e.g. ``"P-PP_1d411b9b" -> "pinky_mcp"``).  Both URDF child and parent
    link names should be included because MuJoCo may invert the kinematic tree.

    MuJoCo ``<body name="...">`` names correspond to original URDF link names.
    Each body may contain a ``<joint ref="...">`` attribute.  The ``ref`` value
    is in the unit set by ``<compiler angle="..."/>`` (radians or degrees).
    """
    tree = ET.parse(mujoco_path)
    mj_root = tree.getroot()

    # Detect angle unit from <compiler angle="..."/>
    compiler_el = mj_root.find("compiler")
    angle_unit = "degree"
    if compiler_el is not None:
        angle_unit = compiler_el.get("angle", "degree")

    ref_offsets = {}
    for body_el in mj_root.iter("body"):
        body_name = body_el.get("name", "")
        bare_joint = link_to_bare_joint.get(body_name)
        if bare_joint is None:
            continue
        joint_el = body_el.find("joint")
        if joint_el is None:
            continue
        ref_val = float(joint_el.get("ref", "0"))
        ref_rad = ref_val if angle_unit == "radian" else math.radians(ref_val)
        ref_offsets[bare_joint] = _normalize_angle(ref_rad)
    ref_offsets["wrist"] = 0.0
    return ref_offsets


def _write_ref_offsets(urdf_path: str, ref_dict: dict) -> str:
    """Write ref offsets JSON sidecar next to the URDF. Returns the sidecar path."""
    sidecar_path = urdf_path + ".ref_offsets.json"
    with open(sidecar_path, "w") as f:
        json.dump(ref_dict, f, indent=2)
    return sidecar_path


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def is_onshape_urdf(urdf_path: str) -> bool:
    """Return True if the URDF uses Onshape-style auto-generated names."""
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    hex_pattern = re.compile(r"_[0-9a-f]{8}$")
    has_hex = False
    has_semantic = False
    for link_el in root.iter("link"):
        name = link_el.get("name", "")
        if hex_pattern.search(name):
            has_hex = True
        if name.startswith(("left_", "right_")):
            has_semantic = True
    return has_hex and not has_semantic


# ---------------------------------------------------------------------------
# Rename
# ---------------------------------------------------------------------------

def _build_onshape_maps(urdf_path: str, hand_type: str):
    """Build link and joint rename maps from an Onshape URDF.

    Returns ``(tree, link_rename, joint_rename)`` where *tree* is the parsed
    ElementTree and the two dicts map original names → semantic names.
    """
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    link_rename = {}
    mr_map = _disambiguate_middle_ring(tree)
    for orig_name, bare_role in mr_map.items():
        link_rename[orig_name] = f"{hand_type}_{bare_role}"
    for link_el in root.iter("link"):
        orig_name = link_el.get("name")
        if orig_name in link_rename:
            continue
        stripped = _strip_hash(orig_name)
        role = _match_link_prefix(stripped)
        if role is not None:
            link_rename[orig_name] = f"{hand_type}_{role}"

    joint_rename = {}
    for joint_el in root.iter("joint"):
        orig_joint_name = joint_el.get("name")
        child_name = joint_el.find("child").get("link")
        parent_name = joint_el.find("parent").get("link")
        child_semantic = link_rename.get(child_name)
        parent_semantic = link_rename.get(parent_name)
        if child_semantic is None or parent_semantic is None:
            continue
        child_bare = child_semantic.split("_", 1)[1]
        parent_bare = parent_semantic.split("_", 1)[1]
        joint_bare = _JOINT_MAP.get((child_bare, parent_bare))
        if joint_bare is not None:
            joint_rename[orig_joint_name] = f"{hand_type}_{joint_bare}"

    return tree, link_rename, joint_rename


def _build_link_to_bare_joint(tree: ET.ElementTree, joint_rename: dict) -> dict:
    """Build original-link-name → bare-joint-name map from an Onshape URDF.

    Includes BOTH child and parent links of each renamed joint so that
    ``_extract_ref_offsets`` works regardless of how the MuJoCo converter
    oriented the kinematic tree.  Child-link mappings take priority.
    """
    root = tree.getroot()
    link_to_bare_joint = {}
    # First pass: parent links (lower priority — may be overwritten)
    for joint_el in root.iter("joint"):
        orig_joint_name = joint_el.get("name")
        if orig_joint_name not in joint_rename:
            continue
        parent_name = joint_el.find("parent").get("link")
        bare_joint = joint_rename[orig_joint_name].split("_", 1)[1]
        link_to_bare_joint[parent_name] = bare_joint
    # Second pass: child links (higher priority — overwrites parent)
    for joint_el in root.iter("joint"):
        orig_joint_name = joint_el.get("name")
        if orig_joint_name not in joint_rename:
            continue
        child_name = joint_el.find("child").get("link")
        bare_joint = joint_rename[orig_joint_name].split("_", 1)[1]
        link_to_bare_joint[child_name] = bare_joint
    return link_to_bare_joint


def rename_urdf(urdf_path: str, hand_type: str) -> None:
    """Rename all links and joints in an Onshape URDF to semantic names.

    Creates a ``.onshape_backup`` copy before modifying the file in-place.
    """
    if hand_type not in ("left", "right"):
        raise ValueError(f"hand_type must be 'left' or 'right', got {hand_type!r}")

    backup_path = urdf_path + ".onshape_backup"
    if not os.path.exists(backup_path):
        shutil.copy2(urdf_path, backup_path)

    tree, link_rename, joint_rename = _build_onshape_maps(urdf_path, hand_type)
    root = tree.getroot()

    # Build link → bare_joint map for MuJoCo ref extraction (before rename)
    link_to_bare_joint = _build_link_to_bare_joint(tree, joint_rename)

    # Extract ref offsets from companion MuJoCo XML (if found)
    mujoco_path = _find_mujoco_xml(urdf_path)
    if mujoco_path is not None:
        ref_offsets = _extract_ref_offsets(mujoco_path, link_to_bare_joint)
        sidecar = _write_ref_offsets(urdf_path, ref_offsets)
        print(f"  Ref offsets sidecar: {sidecar} ({len(ref_offsets)} joints)")
    else:
        print("  No companion MuJoCo XML found — skipping ref offset extraction")

    # Apply renames to the XML
    for link_el in root.iter("link"):
        name = link_el.get("name")
        if name in link_rename:
            link_el.set("name", link_rename[name])

    for joint_el in root.iter("joint"):
        name = joint_el.get("name")
        if name in joint_rename:
            joint_el.set("name", joint_rename[name])
        # Rename parent/child link references
        for tag in ("parent", "child"):
            ref_el = joint_el.find(tag)
            if ref_el is not None:
                ref_name = ref_el.get("link")
                if ref_name in link_rename:
                    ref_el.set("link", link_rename[ref_name])

    tree.write(urdf_path, xml_declaration=False)
    print(f"URDF renamed in-place: {urdf_path}")
    print(f"  Backup saved to: {backup_path}")
    print(f"  Links renamed: {len(link_rename)}")
    print(f"  Joints renamed: {len(joint_rename)}")


# ---------------------------------------------------------------------------
# Main entry point for retargeters
# ---------------------------------------------------------------------------

def load_ref_offsets(urdf_path: str):
    """Load ref offsets from the JSON sidecar next to *urdf_path*.

    Returns the dict if the sidecar exists, otherwise ``None``.
    """
    sidecar_path = urdf_path + ".ref_offsets.json"
    if not os.path.exists(sidecar_path):
        return None
    with open(sidecar_path, "r") as f:
        return json.load(f)


def _generate_ref_sidecar(urdf_path: str, hand_type: str):
    """Try to generate a ref-offsets sidecar for an already-renamed URDF.

    Uses the ``.onshape_backup`` file (original Onshape names) to build the
    link→joint mapping, then finds the companion MuJoCo XML and extracts refs.

    Returns the ref offsets dict on success, or ``None`` on failure.
    """
    backup_path = urdf_path + ".onshape_backup"
    if not os.path.exists(backup_path):
        return None

    mujoco_path = _find_mujoco_xml(urdf_path)
    if mujoco_path is None:
        return None

    try:
        _, _, joint_rename = _build_onshape_maps(backup_path, hand_type)
        tree = ET.parse(backup_path)
        link_to_bare_joint = _build_link_to_bare_joint(tree, joint_rename)
        ref_offsets = _extract_ref_offsets(mujoco_path, link_to_bare_joint)
        sidecar = _write_ref_offsets(urdf_path, ref_offsets)
        print(f"Generated ref offsets sidecar from backup: {sidecar} ({len(ref_offsets)} joints)")
        return ref_offsets
    except Exception as e:
        print(f"Warning: failed to generate ref sidecar from backup: {e}")
        return None


def ensure_semantic_urdf(urdf_path: str, hand_type: str, expected_joint_ids: list):
    """Ensure the URDF at *urdf_path* has semantic joint names.

    If the joints already match *expected_joint_ids* (prefixed with hand_type),
    return immediately. Otherwise, if the file looks like an Onshape export,
    rename it in-place (with backup) and return the same path.

    Also ensures a ``.ref_offsets.json`` sidecar exists next to the URDF.
    If missing, tries to generate one from the ``.onshape_backup`` + companion
    MuJoCo XML.

    Parameters
    ----------
    urdf_path : str
        Path to the URDF file.
    hand_type : str
        ``"left"`` or ``"right"``.
    expected_joint_ids : list[str]
        Bare joint IDs from OrcaHand config (e.g. ``["wrist", "pinky_abd", ...]``).

    Returns
    -------
    dict or None
        Ref offsets dict loaded from the JSON sidecar (if it exists), otherwise
        ``None``.  Callers should fall back to hardcoded ``JOINT_REF_OFFSETS_RAD``
        when ``None`` is returned.
    """
    expected_urdf_names = {f"{hand_type}_{jid}" for jid in expected_joint_ids}

    # Quick check: parse revolute joint names from the URDF
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    urdf_joint_names = set()
    for joint_el in root.iter("joint"):
        if joint_el.get("type") == "revolute":
            urdf_joint_names.add(joint_el.get("name"))

    if not expected_urdf_names == urdf_joint_names:
        if is_onshape_urdf(urdf_path):
            print(f"Detected Onshape URDF naming — auto-renaming joints/links...")
            rename_urdf(urdf_path, hand_type)
        # else: not Onshape and not matching — let the caller's assert handle it

    # Try loading existing sidecar
    ref_offsets = load_ref_offsets(urdf_path)
    if ref_offsets is not None:
        return ref_offsets

    # Sidecar missing — try to generate from backup + MuJoCo XML
    return _generate_ref_sidecar(urdf_path, hand_type)
