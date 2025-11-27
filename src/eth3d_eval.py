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

def evaluatePointCloud(pred_pts, gt_pts, sample_size=5000, K=500, max_eval_dist=0.10):
    """
    Returns dict of metrics:
      - coverage: %% of GT having a predicted point within max_dist
      - mean / median distance on covered points
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
    # 2) Compute alignment
    # -----------------------------
    s, R, t = kabsch_umeyama(pred_corr, gt_corr)
    pred_aligned = (s*(R @ pred_pts.T)).T + t

    # -----------------------------
    # 3) Compute metrics after alignment
    # -----------------------------
    tree_pred = cKDTree(pred_aligned)

    # GT->Pred
    dist_gt_pred, _ = tree_pred.query(gt_pts)
    coverage_mask = dist_gt_pred < max_eval_dist
    coverage = 100.0 * coverage_mask.mean()

    mean_err = dist_gt_pred[coverage_mask].mean() if coverage_mask.any() else np.nan
    med_err = np.median(dist_gt_pred[coverage_mask]) if coverage_mask.any() else np.nan

    return {
        "coverage(%)": coverage,
        "mean_error(m)": mean_err,
        "median_error(m)": med_err,
        "scale": s,
    }
