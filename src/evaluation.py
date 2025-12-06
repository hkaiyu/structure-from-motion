import os

import numpy as np
import pycolmap
from scipy.spatial.transform import Rotation as R
from scipy.spatial import cKDTree

# https://zpl.fi/aligning-point-patterns-with-kabsch-umeyama-algorithm/
def kabsch_umeyama(A, B):
    assert A.shape == B.shape
    n, m = A.shape

    EA = np.mean(A, axis=0)
    EB = np.mean(B, axis=0)
    VarA = np.mean(np.linalg.norm(A - EA, axis=1) ** 2)

    H = ((A - EA).T @ (B - EB)) / n
    U, D, VT = np.linalg.svd(H)
    d = np.sign(np.linalg.det(U) * np.linalg.det(VT))
    S = np.diag([1] * (m - 1) + [d])

    R = U @ S @ VT
    c = VarA / np.trace(np.diag(D) @ S)
    t = EA - c * R @ EB
    return R, c, t

def evaluateAccuracy(sfm, rec_gt, print_summary=False):
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

        # colmap camera center
        C_gt = np.asarray(img_gt.projection_center(), dtype=float).reshape(3)

        # Our camera center
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

    R_sim, s, t_sim = kabsch_umeyama(centers_gt, centers_est)

    # ----- POSE ERRORS -----
    M = np.zeros((3,3))
    for i in range(num_matched):
        Ra = rotations_gt[i]
        Rb = rotations_est[i] @ R_sim.T
        M += Ra @ Rb.T

    U, _, Vt = np.linalg.svd(M)
    Q = U @ Vt

    rot_errors = []
    trans_errors = []

    for i in range(num_matched):
        C_gt = centers_gt[i]
        C_est = centers_est[i]
        R_gt = rotations_gt[i]
        R_est = rotations_est[i]

        C_est_aligned = s * (R_sim @ C_est) + t_sim
        trans_errors.append(float(np.linalg.norm(C_gt - C_est_aligned)))

        R_est_aligned = Q @ (R_est @ R_sim.T)
        R_rel = R_gt @ R_est_aligned.T
        tr = np.clip((np.trace(R_rel) - 1.0) / 2.0, -1.0, 1.0)
        angle_deg = float(np.degrees(np.arccos(tr)))
        rot_errors.append(angle_deg)

    rot_errors = np.asarray(rot_errors, dtype=float)
    trans_errors = np.asarray(trans_errors, dtype=float)

    pose_metrics = {
        "rot_mean_deg": float(rot_errors.mean()),
        "rot_median_deg": float(np.median(rot_errors)),
        "rot_min_deg": float(rot_errors.min()),
        "rot_max_deg": float(rot_errors.max()),
        "trans_mean": float(trans_errors.mean()),
        "trans_median": float(np.median(trans_errors)),
        "trans_min": float(trans_errors.min()),
        "trans_max": float(trans_errors.max()),
        "num_matched_cameras": num_matched,
        "per_camera": [
            {
                "name": matched_names[i],
                "rot_deg": float(rot_errors[i]),
                "trans": float(trans_errors[i])
            }
            for i in range(num_matched)
        ]
    }
    # ----- POINT CLOUD ERRORS -----
    pts_gt = np.asarray([p.xyz for p in rec_gt.points3D.values()], float)
    num_gt_points = pts_gt.shape[0]

    pts_est_aligned = []
    for pt in sfm.pts.values():
        if pt.coord is None:
            continue
        P = np.asarray(pt.coord).reshape(3)
        pts_est_aligned.append(s * (R_sim @ P) + t_sim)
    pts_est_aligned = np.asarray(pts_est_aligned, float)
    num_est_points = pts_est_aligned.shape[0]

    kdt = cKDTree(pts_gt)
    dists, _ = kdt.query(pts_est_aligned, k=1)

    # since we don't have units, we will use scene scale
    # scene size calculated by the size of the bounding box represented by the min colmap cam center to the max
    # colmap cam center
    bbox_min = centers_gt.min(axis=0)
    bbox_max = centers_gt.max(axis=0)
    scene_diag = np.linalg.norm(bbox_max - bbox_min)

    # Completeness thresholds
    th_01pct = 0.001 * scene_diag
    th_05pct = 0.005 * scene_diag
    th_1pct = 0.01 * scene_diag

    completeness_01 = float(np.mean(dists <= th_01pct))
    completeness_05 = float(np.mean(dists <= th_05pct))
    completeness_1 = float(np.mean(dists<= th_1pct))

    pc_metrics = {
        "mean": float(dists.mean()),
        "median": float(np.median(dists)),
        "rmse": float(np.sqrt(np.mean(dists**2))),
        "min": float(dists.min()),
        "max": float(dists.max()),
        "num_gt_points": num_gt_points,
        "num_est_points": num_est_points,
        "scene_diagonal": float(scene_diag),
        "completeness_0_1pct": completeness_01,
        "completeness_0_5pct": completeness_05,
        "completeness_1pct": completeness_1,
    }
    if print_summary:
        print("\n============================================================")
        print("                     SfM ACCURACY SUMMARY")
        print("============================================================")

        print("---- Camera Pose Errors ----")
        print(f"Matched cameras: {num_matched}")
        print(f"Rotation error (degrees): "
              f"mean={pose_metrics['rot_mean_deg']:.3f}, "
              f"median={pose_metrics['rot_median_deg']:.3f}, "
              f"min={pose_metrics['rot_min_deg']:.3f}, "
              f"max={pose_metrics['rot_max_deg']:.3f}")
        print(f"Translation error: " 
              f"mean={pose_metrics['trans_mean']:.4f}, "
              f"median={pose_metrics['trans_median']:.4f}, "
              f"min={pose_metrics['trans_min']:.4f}, "
              f"max={pose_metrics['trans_max']:.4f}")

        print("Per-camera pose error:")
        per_cam = pose_metrics['per_camera']
        for entry in per_cam:
            print(f"\tName: {entry['name']}")
            print(f"\tRotation error (degrees): {entry['rot_deg']:.4f}")
            print(f"\tTranslation error (unknown units): {entry['trans']:.4f}")

        print("\n---- Point Cloud Errors ----")
        print(f"GT points: {num_gt_points}")
        print(f"Est points: {num_est_points}")
        print(f"Scene diagonal: {scene_diag:.4f} (units)")

        print(f"Dist error mean: {pc_metrics['mean']:.4f}")
        print(f"Dist error median:{pc_metrics['median']:.4f}")
        print(f"Dist error RMSE: {pc_metrics['rmse']:.4f}")
        print(f"Dist error min: {pc_metrics['min']:.4f}")
        print(f"Dist error max: {pc_metrics['max']:.4f}")

        print("\n---- Completeness ----")
        print("Since we have no ground truth measurements, we cannot measure completeness in sense of "
              "the number of points in our point cloud that are within x meters of the nearest GT point cloud. "
              "Instead, we will measure completeness as the number of points in our point cloud within x% of the size "
              "of the scene, where the scene size is the bounding box from the smallest GT camera center to largest."
              )
        print(f"Completeness (0.1% scene size): {completeness_01*100:.2f}%")
        print(f"Completeness (0.5% scene size): {completeness_05*100:.2f}%")
        print(f"Completeness (1.0% scene size): {completeness_1*100:.2f}%")

        print("\n======================================\n")

    return {
        "pose_errors": pose_metrics,
        "pc_errors": pc_metrics,
        "scale": s,
        "rotation": R_sim,
        "translation": t_sim,
        "aligned_pts": pts_est_aligned,
        "point_errors": dists
    }
