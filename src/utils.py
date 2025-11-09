import cv2 as cv
import numpy as np
import os


def estimatePose(pts1, pts2, E, K):
  # calculates _relative_ pose
  _, R, t, mask = cv.recoverPose(E, pts1, pts2, K)
  return R, t, mask


def triangulate(pts1, pts2, R1, t1, R2, t2, K):
  # Projection matrix = K * [R | t]
  P1 = (K @ np.hstack((R1, t1))).astype(np.float32) # 3x4
  P2 = (K @ np.hstack((R2, t2))).astype(np.float32) # 3x4
  pts_h = cv.triangulatePoints(P1, P2, pts1.T, pts2.T)
  return (pts_h[:3] / pts_h[3]).T # homogeneous -> euclidean coords, (N, 3)


def createIntrinsicsMatrix(fx, fy, cx, cy, s=0):
    """
    Creates intrinsics matrix (K) such that:
        - fx = horizontal focal length
        - fy = vertical focal length
        - (cx, cy) = principal point
        - s = skew factor
    """
    return np.array([[fx, s, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)


def load_all_images(dataset_path):
    """
    Load all images from a dataset directory.

    Args:
        dataset_path (str or os.PathLike): Directory containing images.

    Returns:
        list[np.ndarray]: List of images in BGR order as loaded by OpenCV, sorted by filename.
    """
    if isinstance(dataset_path, os.PathLike):
        dataset_path = os.fspath(dataset_path)

    valid_exts = (".jpg", ".jpeg", ".png", ".bmp")
    try:
        files = sorted(
            [f for f in os.listdir(dataset_path) if f.lower().endswith(valid_exts)]
        )
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")

    images = []
    for fname in files:
        fpath = os.path.join(dataset_path, fname)
        img = cv.imread(fpath)
        if img is None:
            print(f"Warning: Failed to read image: {fpath}")
            continue
        images.append(img)
    return images
