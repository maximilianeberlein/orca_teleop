"""Multi-camera triangulation using weighted DLT.

Follows the patterns established by:
- aniposelib (lambdaloop/aniposelib): NaN sentinel, min-cameras check, SVD-based DLT
- Pose2Sim (perfanalytics/pose2sim): confidence-weighted DLT rows, iterative camera
  exclusion based on reprojection error
"""

import numpy as np


def triangulate_points(projection_matrices, points_2d, confidences=None):
    """Triangulate N 3D points from M camera views using weighted DLT.

    Uses batch SVD when all cameras have positive confidence (fast path),
    falling back to per-point loop for points with masked-out cameras.

    Args:
        projection_matrices: list of M (3, 4) projection matrices
        points_2d: (M, N, 2) array of 2D pixel coordinates per camera
        confidences: optional (M, N) array of per-landmark confidence weights.
            Cameras with confidence <= 0 are excluded for that landmark.

    Returns:
        (N, 3) array of triangulated 3D points
    """
    M = len(projection_matrices)
    N = points_2d.shape[1]
    if confidences is None:
        confidences = np.ones((M, N), dtype=np.float64)

    P = np.array(projection_matrices)  # (M, 3, 4)
    points_3d = np.full((N, 3), np.nan, dtype=np.float64)

    # Determine which points have all cameras active (fast path)
    all_active = np.all(confidences > 0, axis=0)  # (N,)
    batch_indices = np.where(all_active)[0]
    fallback_indices = np.where(~all_active)[0]

    # Fast path: batch DLT for points where all M cameras contribute
    if len(batch_indices) > 0:
        Nb = len(batch_indices)
        # Build A matrix: (Nb, 2*M, 4)
        A = np.empty((Nb, 2 * M, 4), dtype=np.float64)
        for i in range(M):
            u = points_2d[i, batch_indices, 0]  # (Nb,)
            v = points_2d[i, batch_indices, 1]  # (Nb,)
            w = confidences[i, batch_indices]    # (Nb,)
            # row_u = w * (u * P[i, 2] - P[i, 0])  → (Nb, 4)
            A[:, 2 * i, :] = w[:, None] * (u[:, None] * P[i, 2:3, :] - P[i, 0:1, :])
            A[:, 2 * i + 1, :] = w[:, None] * (v[:, None] * P[i, 2:3, :] - P[i, 1:2, :])

        _, _, Vt = np.linalg.svd(A)  # Vt: (Nb, 4, 4)
        X = Vt[:, -1, :]  # (Nb, 4) — last row of Vt for each point
        valid = np.abs(X[:, 3]) > 1e-10
        points_3d[batch_indices[valid]] = X[valid, :3] / X[valid, 3:4]

    # Fallback: per-point loop for points with some cameras masked out
    for j in fallback_indices:
        rows = []
        for i in range(M):
            w = confidences[i, j]
            if w <= 0:
                continue
            u, v = points_2d[i, j, 0], points_2d[i, j, 1]
            rows.append(w * (u * P[i, 2] - P[i, 0]))
            rows.append(w * (v * P[i, 2] - P[i, 1]))

        if len(rows) < 4:  # need at least 2 cameras (4 rows)
            continue

        A_single = np.array(rows)
        _, s, Vt = np.linalg.svd(A_single)
        X = Vt[-1]
        if abs(X[3]) > 1e-10:
            points_3d[j] = X[:3] / X[3]

    return points_3d


def compute_reprojection_error(P, point_3d, point_2d):
    """Compute reprojection error (pixels) for a single point."""
    X_hom = np.append(point_3d, 1.0)
    projected = P @ X_hom
    if abs(projected[2]) < 1e-10:
        return np.inf
    projected_2d = projected[:2] / projected[2]
    return np.linalg.norm(projected_2d - point_2d)


def triangulate_with_refinement(projection_matrices, points_2d, confidences,
                                reproj_threshold=10.0):
    """Two-pass weighted DLT with reprojection-error-based camera exclusion.

    Matches the Pose2Sim pattern: triangulate, check reprojection error per
    camera, exclude the worst camera if error > threshold, re-triangulate.

    Returns:
        points_3d: (N, 3) triangulated points
        quality: (N,) mean reprojection error per point (across used cameras)
    """
    M = len(projection_matrices)
    N = points_2d.shape[1]

    P = np.array(projection_matrices)  # (M, 3, 4)
    points_3d = triangulate_points(projection_matrices, points_2d, confidences)
    quality = np.full(N, np.nan, dtype=np.float64)

    # Find valid (non-NaN) points
    valid_mask = ~np.any(np.isnan(points_3d), axis=1)  # (N,)
    valid_indices = np.where(valid_mask)[0]

    if len(valid_indices) == 0:
        return points_3d, quality

    # Batch reprojection error for all valid points across all cameras
    valid_pts = points_3d[valid_indices]  # (Nv, 3)
    X_hom = np.hstack([valid_pts, np.ones((len(valid_pts), 1))])  # (Nv, 4)

    # errors_all: (M, Nv) reprojection errors
    errors_all = np.full((M, len(valid_indices)), np.inf, dtype=np.float64)
    active_all = confidences[:, valid_indices] > 0  # (M, Nv)

    for i in range(M):
        cam_active = active_all[i]  # (Nv,)
        if not np.any(cam_active):
            continue
        projected = (P[i] @ X_hom.T).T  # (Nv, 3)
        safe_z = np.where(np.abs(projected[:, 2]) > 1e-10, projected[:, 2], 1e-10)
        proj_2d = projected[:, :2] / safe_z[:, None]  # (Nv, 2)
        err = np.linalg.norm(proj_2d - points_2d[i, valid_indices], axis=1)  # (Nv,)
        errors_all[i, cam_active] = err[cam_active]

    n_active_per_point = np.sum(active_all, axis=0)  # (Nv,)

    # Compute mean error for active cameras
    for k, j in enumerate(valid_indices):
        if n_active_per_point[k] < 2:
            continue
        active_mask = active_all[:, k]
        quality[j] = np.mean(errors_all[active_mask, k])

        # Camera exclusion: remove worst camera if error > threshold and >2 cameras
        worst_idx = np.argmax(errors_all[:, k] * active_mask)
        if errors_all[worst_idx, k] > reproj_threshold and n_active_per_point[k] > 2:
            refined_confs = confidences[:, j].copy()
            refined_confs[worst_idx] = 0.0
            single_2d = points_2d[:, j:j+1, :]
            single_confs = refined_confs.reshape(M, 1)
            refined_pt = triangulate_points(
                projection_matrices, single_2d, single_confs
            )
            if not np.any(np.isnan(refined_pt[0])):
                points_3d[j] = refined_pt[0]
                re_errors = []
                for i in range(M):
                    if refined_confs[i] > 0:
                        re_errors.append(compute_reprojection_error(
                            projection_matrices[i], points_3d[j], points_2d[i, j]
                        ))
                if re_errors:
                    quality[j] = np.mean(re_errors)

    return points_3d, quality
