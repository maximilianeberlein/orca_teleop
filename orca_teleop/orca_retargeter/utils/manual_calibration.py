import numpy as np
from scipy.spatial.transform import Rotation


def apply_manual_calibration(points, center, scale=1.0, rotation_deg=None, translation=None):
    """Apply manual calibration transform to MANO points in URDF world space.

    Scales relative to center, rotates relative to center (extrinsic XYZ Euler),
    then translates in world space.
    """
    if rotation_deg is None:
        rotation_deg = np.zeros(3)
    if translation is None:
        translation = np.zeros(3)

    no_scale = abs(scale - 1.0) < 1e-6
    no_rotation = np.all(np.abs(rotation_deg) < 1e-6)
    no_translation = np.all(np.abs(translation) < 1e-6)
    if no_scale and no_rotation and no_translation:
        return points

    result = points - center
    if not no_scale:
        result = result * scale
    if not no_rotation:
        rot = Rotation.from_euler('XYZ', rotation_deg, degrees=True)
        result = result @ rot.as_matrix().T
    result = result + center
    if not no_translation:
        result = result + translation
    return result
