import numpy as np
from scipy.spatial import cKDTree

# code from https://zpl.fi/aligning-point-patterns-with-kabsch-umeyama-algorithm/
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

    return c, R, t

def _cam_center_from_Rt(R, t):
    # Ensure column vector
    if t.ndim == 1:
        t = t.reshape(3, 1)
    return (-R.T @ t).reshape(3)

def _rotation_error_deg(R1, R2):
    # Geodesic distance on SO(3)
    R_rel = R2 @ R1.T
    c = (np.trace(R_rel) - 1.0) / 2.0
    c = np.clip(c, -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))

def evaluatePointCloud(pred_pts, gt_pts, pred_cams=None, gt_cams=None, sample_size=5000, K=500, max_eval_dist=0.10):
    """
    Returns dict of metrics:
      - coverage: %% of GT having a predicted point within max_dist
      - mean / median distance on covered points
      - optional camera pose metrics if pred_cams and gt_cams are provided:
          * registered camera fraction
          * translation and rotation error stats for cameras present in both
          * list of GT cameras missing from the prediction
    """

    pred_pts = np.asarray(pred_pts, float)
    gt_pts = np.asarray(gt_pts, float)

    if len(pred_pts) < 20:
        return {"error": "Too few predicted points"}

    print(f"[EVAL] pred: {len(pred_pts)} pts, GT: {len(gt_pts)} pts")

    # -----------------------------
    # 1) Coarse matching (Pred -> GT)
    # -----------------------------
    tree_gt = cKDTree(gt_pts)

    sample_size = min(sample_size, len(pred_pts))
    sample_idx = np.random.choice(len(pred_pts), size=sample_size, replace=False)
    pred_sample = pred_pts[sample_idx]

    dist_init, idx_init = tree_gt.query(pred_sample, k=1)

    # Pick best-K matches for initial alignment
    K = min(K, len(pred_sample))
    best = np.argsort(dist_init)[:K]

    pred_corr = pred_sample[best]
    gt_corr = gt_pts[idx_init[best]]

    if len(pred_corr) < 10:
        return {"error": "No overlap for coarse alignment"}

    print(f"[EVAL] Using {len(pred_corr)} coarse correspondences")

    # -----------------------------
    # 2) Compute alignment (similarity: scale s, rotation R_align, translation t_align)
    # -----------------------------
    s, R_align, t_align = kabsch_umeyama(pred_corr, gt_corr)
    pred_aligned = (s * (R_align @ pred_pts.T)).T + t_align

    # -----------------------------
    # 3) Compute point-cloud metrics after alignment
    # -----------------------------
    tree_pred = cKDTree(pred_aligned)

    # GT->Pred
    dist_gt_pred, _ = tree_pred.query(gt_pts)
    coverage_mask = dist_gt_pred < max_eval_dist
    coverage = 100.0 * coverage_mask.mean()

    mean_err = dist_gt_pred[coverage_mask].mean() if coverage_mask.any() else np.nan
    med_err = np.median(dist_gt_pred[coverage_mask]) if coverage_mask.any() else np.nan

    results = {
        "coverage(%)": coverage,
        "mean_error(m)": mean_err,
        "median_error(m)": med_err,
        "scale": s,
    }

    # -----------------------------
    # 4) Optional: camera pose metrics (after applying same similarity transform)
    # -----------------------------
    if pred_cams is not None and gt_cams is not None and len(pred_cams) > 0 and len(gt_cams) > 0:
        pred_keys = set(pred_cams.keys())
        gt_keys = set(gt_cams.keys())
        common = sorted(list(pred_keys & gt_keys))

        missing_gt = sorted(list(gt_keys - pred_keys))
        results["cam_registered_frac(%)"] = 100.0 * (len(common) / max(1, len(gt_keys)))
        if missing_gt:
            # include a small sample to keep logs readable
            results["missing_gt_cameras"] = missing_gt[:50]

        trans_errs = []
        rot_errs = []
        per_cam = []

        for name in common:
            R_p, t_p = pred_cams[name]
            R_g, t_g = gt_cams[name]

            # Camera centers
            C_p = _cam_center_from_Rt(R_p, t_p)
            C_g = _cam_center_from_Rt(R_g, t_g)

            # Apply similarity to predicted center
            C_p_aligned = s * (R_align @ C_p) + t_align

            # Rotation alignment to GT frame: express predicted rotation in GT world
            R_p_aligned = R_p @ R_align.T

            # Errors
            te = np.linalg.norm(C_p_aligned - C_g)
            re = _rotation_error_deg(R_p_aligned, R_g)

            trans_errs.append(te)
            rot_errs.append(re)
            per_cam.append({"id": name, "trans_err": float(te), "rot_err_deg": float(re)})

        if len(trans_errs) > 0:
            trans_errs = np.asarray(trans_errs, float)
            rot_errs = np.asarray(rot_errs, float)

            results.update({
                "cam_trans_mean(m)": float(np.mean(trans_errs)),
                "cam_rot_mean(deg)": float(np.mean(rot_errs)),
                "per_camera_errors": sorted(per_cam, key=lambda x: x["id"]),
            })

    return results
