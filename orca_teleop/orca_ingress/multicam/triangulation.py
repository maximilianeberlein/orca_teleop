"""Multi-camera triangulation using weighted DLT.

Follows the patterns established by:
- aniposelib (lambdaloop/aniposelib): NaN sentinel, min-cameras check, SVD-based DLT
- Pose2Sim (perfanalytics/pose2sim): confidence-weighted DLT rows, iterative camera
  exclusion based on reprojection error
"""

import numpy as np


def triangulate_points(projection_matrices, points_2d, confidences=None):
    """Triangulate N 3D points from M camera views using weighted DLT.

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

    points_3d = np.full((N, 3), np.nan, dtype=np.float64)

    for j in range(N):
        rows = []
        for i in range(M):
            w = confidences[i, j]
            if w <= 0:
                continue
            P = projection_matrices[i]
            u, v = points_2d[i, j, 0], points_2d[i, j, 1]
            rows.append(w * (u * P[2] - P[0]))
            rows.append(w * (v * P[2] - P[1]))

        if len(rows) < 4:  # need at least 2 cameras (4 rows)
            continue

        A = np.array(rows)
        _, s, Vt = np.linalg.svd(A)
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

    points_3d = triangulate_points(projection_matrices, points_2d, confidences)
    quality = np.full(N, np.nan, dtype=np.float64)

    for j in range(N):
        if np.any(np.isnan(points_3d[j])):
            continue

        errors = np.full(M, np.inf, dtype=np.float64)
        active = np.zeros(M, dtype=bool)
        for i in range(M):
            if confidences[i, j] <= 0:
                continue
            active[i] = True
            errors[i] = compute_reprojection_error(
                projection_matrices[i], points_3d[j], points_2d[i, j]
            )

        n_active = np.sum(active)
        if n_active < 2:
            continue

        quality[j] = np.mean(errors[active])

        # Iterative camera exclusion (Pose2Sim pattern):
        # remove the worst camera if its error exceeds threshold
        worst_idx = np.argmax(errors * active)
        if errors[worst_idx] > reproj_threshold and n_active > 2:
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
