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
    C1 = -R1.T @ t1
    C2 = -R2.T @ t2
    v1 = (X.reshape(3) - C1.squeeze())
    v2 = (X.reshape(3) - C2.squeeze())
    v1 /= np.linalg.norm(v1)
    v2 /= np.linalg.norm(v2)
    # compute angle
    dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
    return np.degrees(np.arccos(dot))

def estimatePose(pts1, pts2, E, K):
    # calculates _relative_ pose
    _, R, t, mask = cv.recoverPose(E, pts1, pts2, K)
    return R, t, mask

def triangulate(pts1, pts2, R1, t1, R2, t2, K, decimal_pts=2):
    # Projection matrix = K * [R | t]
    P1 = (K @ np.hstack((R1, t1))).astype(np.float32) # 3x4
    P2 = (K @ np.hstack((R2, t2))).astype(np.float32) # 3x4
    pts_h = cv.triangulatePoints(P1, P2, pts1.T, pts2.T)
    pts_3d = (pts_h[:3] / pts_h[3]).T # homogeneous -> euclidean coords, (N, 3)
    pts_3d = np.round(pts_3d, decimals=decimal_pts)
    return pts_3d

# per-point variant of triangulate()
def triangulatePoint(pt1, pt2, R1, t1, R2, t2, K):
    P1 = (K @ np.hstack((R1, t1))).astype(np.float32)
    P2 = (K @ np.hstack((R2, t2))).astype(np.float32)
    pt1 = np.array(pt1, dtype=np.float32).reshape(2, 1)
    pt2 = np.array(pt2, dtype=np.float32).reshape(2, 1)
    X_h = cv.triangulatePoints(P1, P2, pt1, pt2)
    X = (X_h[:3] / X_h[3]).reshape(3)
    return X.astype(np.float32)

def projectPoint(K, R, t, X):
    X_cam = R @ X.reshape(3, 1) + t
    x = K @ X_cam
    x = x.ravel()
    if x[2] == 0:
        return None
    return x[:2] / x[2]

def reprojectionError(K, R, t, X, u):
    u_hat = projectPoint(K, R, t, X)
    if u_hat is None:
        return np.inf
    return np.linalg.norm(u_hat - u)

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
