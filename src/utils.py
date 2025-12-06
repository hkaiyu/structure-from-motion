import os
from time import time

import cv2 as cv
import numpy as np

# adapted from https://www.geeksforgeeks.org/python/timing-functions-with-decorators-python/
def profile(func):
    """
    Decorator to time functions.
    Example usage:

    @profile
    def long_time(n):
        for i in range(n):
            for j in range(100000):
                i*j
    """
    def wrap_func(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        print(f'[PERF] Function {func.__name__!r} executed in {(t2-t1):.4f}s')
        return result
    return wrap_func

# chierality check
def pointDepth(R, t, X):
    X_cam = R @ X.reshape(3,1) + t
    return X_cam[2,0] # >0 means in front of cam

# parallax check; low angle -> do not use
def parallaxAngle(R1, t1, R2, t2, X):
    # Compute viewing rays from each camera toward the triangulated point
    X = np.asarray(X, dtype=float).reshape(3)
    C1 = -R1.T @ t1.reshape(3)
    C2 = -R2.T @ t2.reshape(3)

    v1 = X - C1
    v2 = X - C2
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-8 or n2 < 1e-8:
        return 0.0

    cos_theta = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))

def estimatePose(pts1, pts2, E, K):
    # calculates _relative_ pose
    _, R, t, mask = cv.recoverPose(E, pts1, pts2, K)
    return R, t, mask

def triangulate(pts1, pts2, R1, t1, R2, t2, K):
    t1 = t1.reshape(3, 1)
    t2 = t2.reshape(3, 1)

    P1 = (K @ np.hstack([R1, t1])).astype(np.float32)
    P2 = (K @ np.hstack([R2, t2])).astype(np.float32)
    pts_h = cv.triangulatePoints(P1, P2, pts1.T, pts2.T)
    pts_3d = (pts_h[:3] / pts_h[3]).T
    return pts_3d.astype(np.float32)

# per-point variant of triangulate()
def triangulatePoint(pt1, pt2, R1, t1, R2, t2, K):
    t1 = t1.reshape(3, 1)
    t2 = t2.reshape(3, 1)
    P1 = (K @ np.hstack((R1, t1))).astype(np.float32)
    P2 = (K @ np.hstack((R2, t2))).astype(np.float32)
    x1, y1 = pt1
    x2, y2 = pt2

    A = np.zeros((4, 4))
    A[0] = x1 * P1[2] - P1[0]
    A[1] = y1 * P1[2] - P1[1]
    A[2] = x2 * P2[2] - P2[0]
    A[3] = y2 * P2[2] - P2[1]

    _, _, Vt = np.linalg.svd(A)
    X_h = Vt[-1]
    X_h /= X_h[3]
    return X_h[:3]

def reprojectionError(K, R, t, X, u):
    X = np.asarray(X, dtype=float).reshape(3, 1)
    u = np.asarray(u, dtype=float).reshape(2)

    x_cam = R @ X + t.reshape(3, 1)
    if x_cam[2, 0] <= 0:
        return 1e9 #behind cam

    x_norm = x_cam[:2, 0] / x_cam[2, 0]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    u_proj = np.array([fx * x_norm[0] + cx, fy * x_norm[1] + cy])
    return float(np.linalg.norm(u_proj - u))

@profile
def loadAllImages(image_dir):
    exts = (".jpg", ".jpeg", ".png", ".gif", ".bmp")
    images = []
    for fname in os.listdir(image_dir):
        if fname.lower().endswith(exts):
            fp = os.path.join(image_dir, fname)
            img = cv.imread(fp)
            if img is None:
                print(f"Warning: Failed to read image: {fpath}")
                continue
            images.append((fname, img))
    return images

# ---------------- LaTeX table utilities ---------------- #

TABLE_SCHEMAS = {
    "cameraPoseErrors": {
        "headers": [
            "Rot mean (deg)", "Rot median (deg)", "Rot min", "Rot max",
            "Trans mean", "Trans median", "Trans min", "Trans max",
            "Matched cams"
        ],
        "caption": "Camera pose error summary",
        "label": "tab:camera_pose_errors",
    },
    "pointCloudErrors": {
        "headers": [
            "GT points", "Est points", "Scene diag",
            "Compl 0.1%", "Compl 0.5%", "Compl 1.0%", "Runtime (s)"
        ],
        "caption": "Point cloud quality metrics",
        "label": "tab:point_cloud_errors",
    },
    "projectionErrors": {
        "headers": [
            "GT dist min", "GT dist max",
            "Log GT min", "Log GT max",
            "Reproj min (px)", "Reproj max (px)"
        ],
        "caption": "Projection and reprojection error ranges",
        "label": "tab:projection_errors",
    },
}


def getTableHeaders(table_type: str):
    spec = TABLE_SCHEMAS.get(table_type)
    if not spec:
        raise ValueError(f"Unknown table type: {table_type}")
    return ["Dataset"] + spec["headers"]


def appendToDataTable(table_name: str, data_row: str, base_dir: str = None):
    """
    Append a single formatted row to report/tables/<table_name>.txt.
    Ensures directory exists and that rows end with LaTeX linebreak
    """
    if base_dir is None:
        base_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "report", "tables")
    os.makedirs(base_dir, exist_ok=True)
    # ensure row ends with ' \\' (LaTeX newline)
    row = data_row.rstrip()
    if not row.endswith("\\\\"):
        row = row + " \\\\"  # literal \\ in output
    fpath = os.path.join(base_dir, f"{table_name}.txt")
    with open(fpath, "a", encoding="utf-8") as f:
        f.write(row + "\n")


def printLatexTable(data: str, table_type: str, caption: str = None, label: str = None) -> str:
    """
    Wrap raw LaTeX table body lines (e.g., 'dataset & v1 & v2 \\') with a minimal LaTeX table shell.
    Returns the full LaTeX string.
    """
    spec = TABLE_SCHEMAS.get(table_type)
    if not spec:
        raise ValueError(f"Unknown table type: {table_type}")

    headers = ["Dataset"] + spec["headers"]
    if caption is None:
        caption = spec["caption"]
    if label is None:
        label = spec["label"]

    # Column alignment: l for dataset, centered for the rest
    col_align = "l" + ("c" * (len(headers) - 1))

    begin = (
        "\\begin{table}[ht]\n"
        "\\centering\n"
        "\\small\n"
        f"\\begin{{tabular}}{{{col_align}}}\n"
        "\\hline\n"
        + " & ".join(headers)
        + " \\\\\
"
        "\\hline\n"
    )

    # Ensure data has trailing newline
    body = data.strip() + "\n"

    end = (
        "\\hline\n"
        "\\end{tabular}\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        "\\end{table}\n"
    )
    return begin + body + end
