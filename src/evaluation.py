import os

import numpy as np
import pycolmap
from scipy.spatial.transform import Rotation as R
from scipy.spatial import cKDTree

def umeyama(X, Y):
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    assert X.shape == Y.shape and X.shape[1] == 3

    n = X.shape[0]
    if n < 3:
        s = 1.0
        R = np.eye(3, dtype=float)
        t = np.zeros(3, dtype=float)
        return s, R, t

    mu_X = X.mean(axis=0)
    mu_Y = Y.mean(axis=0)

    X0 = X - mu_X
    Y0 = Y - mu_Y
    sigma = (Y0.T @ X0) / n

    U, D, Vt = np.linalg.svd(sigma)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1.0

    R = U @ S @ Vt
    var_Y = (Y0**2).sum() / n
    s = np.trace(np.diag(D) @ S) / var_Y
    t = mu_X - s * (R @ mu_Y)

    return float(s), R, t


def evaluateAccuracy(sfm, rec_gt):
    gt_by_name = {}
    for img_id, img_gt in rec_gt.images.items():
        basename = os.path.basename(img_gt.name)
        gt_by_name[basename] = img_gt

    centers_gt = []
    centers_est = []
    rotations_gt = []
    rotations_est = []
    matched_names = []

    for img in sfm.images.values():
        if getattr(img, "R", None) is None or getattr(img, "t", None) is None:
            continue

        if getattr(img, "path", None) is None:
            continue

        basename = os.path.basename(img.path)
        img_gt = gt_by_name.get(basename, None)
        if img_gt is None or not img_gt.has_pose:
            continue

        C_gt = np.asarray(img_gt.projection_center(), dtype=float).reshape(3)

        # Our camera center (assuming x_cam = R * x_world + t)
        R_est = np.asarray(img.R, dtype=float).reshape(3, 3)
        t_est = np.asarray(img.t, dtype=float).reshape(3)
        C_est = -R_est.T @ t_est

        # GT rotation matrix (world -> cam)
        cam_from_world = img_gt.cam_from_world()
        R_gt = cam_from_world.rotation.matrix()

        centers_gt.append(C_gt)
        centers_est.append(C_est)
        rotations_gt.append(R_gt)
        rotations_est.append(R_est)
        matched_names.append(basename)

    centers_gt = np.asarray(centers_gt, dtype=float)
    centers_est = np.asarray(centers_est, dtype=float)
    rotations_gt = np.asarray(rotations_gt, dtype=float)
    rotations_est = np.asarray(rotations_est, dtype=float)

    num_matched = centers_gt.shape[0]

    if num_matched == 0:
        pose_metrics = {
            "rot_mean_deg": None,
            "rot_median_deg": None,
            "trans_mean": None,
            "trans_median": None,
            "num_matched_cameras": 0,
        }
        pc_metrics = {
            "pc_mean": None,
            "pc_median": None,
            "pc_rmse": None,
            "pc_min": None,
            "pc_max": None,
            "num_gt_points": len(rec_gt.points3D),
            "num_est_points": len(sfm.pts),
        }
        return {
            "pose_errors": pose_metrics,
            "pc_errors": pc_metrics,
            "scale": 1.0,
            "rotation": np.eye(3, dtype=float),
            "translation": np.zeros(3, dtype=float),
        }

    s, R_sim, t_sim = umeyama(centers_gt, centers_est)

    # ----- POSE ERRORS -----
    rot_errors = []
    trans_errors = []

    for i in range(num_matched):
        C_gt = centers_gt[i]
        C_est = centers_est[i]
        R_gt = rotations_gt[i]
        R_est = rotations_est[i]

        # Transform our camera center by Sim(3): C_aligned = s * R_sim * C_est + t
        C_est_aligned = s * (R_sim @ C_est) + t_sim

        # Translation error as distance between centers
        trans_errors.append(float(np.linalg.norm(C_gt - C_est_aligned)))

        # To align rotations: world is rotated by R_sim, so camera rotations transform as:
        R_est_aligned = R_est @ R_sim.T

        # Relative rotation: R_rel = R_gt * R_est_aligned^T
        R_rel = R_gt @ R_est_aligned.T
        tr = np.clip((np.trace(R_rel) - 1.0) / 2.0, -1.0, 1.0)
        angle_deg = float(np.degrees(np.arccos(tr)))
        rot_errors.append(angle_deg)

    rot_errors = np.asarray(rot_errors, dtype=float)
    trans_errors = np.asarray(trans_errors, dtype=float)

    pose_metrics = {
        "rot_mean_deg": float(rot_errors.mean()) if rot_errors.size > 0 else None,
        "rot_median_deg": float(np.median(rot_errors)) if rot_errors.size > 0 else None,
        "trans_mean": float(trans_errors.mean()) if trans_errors.size > 0 else None,
        "trans_median": float(np.median(trans_errors)) if trans_errors.size > 0 else None,
        "num_matched_cameras": int(num_matched),
    }

    # ----- POINT CLOUD ERRORS -----
    pts_gt = []
    for p in rec_gt.points3D.values():
        pts_gt.append(np.asarray(p.xyz, dtype=float).reshape(3))
    pts_gt = np.asarray(pts_gt, dtype=float)
    num_gt_points = pts_gt.shape[0]

    pts_est_aligned = []
    for pt in sfm.pts.values():
        if getattr(pt, "coord", None) is None:
            continue
        P = np.asarray(pt.coord, dtype=float).reshape(3)
        P_aligned = s * (R_sim @ P) + t_sim
        pts_est_aligned.append(P_aligned)
    pts_est_aligned = np.asarray(pts_est_aligned, dtype=float)
    num_est_points = pts_est_aligned.shape[0]

    if num_gt_points > 0 and num_est_points > 0:
        kdt = cKDTree(pts_gt)
        dists, _ = kdt.query(pts_est_aligned, k=1)

        pc_metrics = {
            "pc_mean": float(dists.mean()),
            "pc_median": float(np.median(dists)),
            "pc_rmse": float(np.sqrt(np.mean(dists**2))),
            "pc_min": float(dists.min()),
            "pc_max": float(dists.max()),
            "num_gt_points": int(num_gt_points),
            "num_est_points": int(num_est_points),
        }
    else:
        pc_metrics = {
            "pc_mean": None,
            "pc_median": None,
            "pc_rmse": None,
            "pc_min": None,
            "pc_max": None,
            "num_gt_points": int(num_gt_points),
            "num_est_points": int(num_est_points),
        }

    return {
        "pose_errors": pose_metrics,
        "pc_errors": pc_metrics,
        "scale": float(s),
        "rotation": R_sim,
        "translation": t_sim,
    }
