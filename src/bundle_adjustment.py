import numpy as np
import cv2 as cv
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from utils import profile

# Code from https://scipy-cookbook.readthedocs.io/items/bundle_adjustment.html
def rotate(points, rot_vecs):
    """Rotate points by given rotation vectors."""
    theta = np.linalg.norm(rot_vecs, axis=1)[:, np.newaxis]
    with np.errstate(invalid='ignore'):
        v = rot_vecs / theta
        v = np.nan_to_num(v)
    dot = np.sum(points * v, axis=1)[:, np.newaxis]
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    return cos_theta * points + sin_theta * np.cross(v, points) + dot * (1 - cos_theta) * v

# Adapted from https://scipy-cookbook.readthedocs.io/items/bundle_adjustment.html
def project(points, camera_poses, K):
    """Convert 3-D points to 2-D by projecting onto images."""

    rvecs = camera_poses[:, :3]
    tvecs = camera_poses[:, 3:]

    pts_proj = rotate(points, rvecs) + tvecs

    x = pts_proj[:, 0] / pts_proj[:, 2]
    y = pts_proj[:, 1] / pts_proj[:, 2]

    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    u = fx * x + cx
    v = fy * y + cy
    return np.vstack([u, v]).T

# Adapted from https://scipy-cookbook.readthedocs.io/items/bundle_adjustment.html
def baResiduals(params, n_cameras, n_points, camera_indices, point_indices, points_2d, K):
    """Compute residuals.
    
    `params` contains camera parameters and 3-D coordinates.
    """
    camera_params = params[:n_cameras * 6].reshape((n_cameras, 6))
    points_3d = params[n_cameras * 6:].reshape((n_points, 3))
    cam_obs = camera_params[camera_indices]
    pts_obs = points_3d[point_indices]
    pts_proj = project(pts_obs, cam_obs, K) 
    return (pts_proj - points_2d).ravel()

# Adapted from https://scipy-cookbook.readthedocs.io/items/bundle_adjustment.html
def baSparsity(n_cameras, n_points, camera_indices, point_indices):
    m = camera_indices.size * 2         # 2 residuals (u, v) per observation
    n = n_cameras * 6 + n_points * 3    # 6 params per camera, 3 per point

    A = lil_matrix((m, n), dtype=int)
    obs_idx = np.arange(camera_indices.size)

    # Camera parameter blocks
    for s in range(6):
        A[2 * obs_idx, camera_indices * 6 + s] = 1
        A[2 * obs_idx + 1, camera_indices * 6 + s] = 1

    # Point parameter blocks
    base = n_cameras * 6
    for s in range(3):
        A[2 * obs_idx, base + point_indices * 3 + s] = 1
        A[2 * obs_idx + 1, base + point_indices * 3 + s] = 1

    return A

def baBuildProblem(sfm):
    """ Build camera_params, points_3d, camera_indices, point_indices, points_2d from SfmData. """
    # map image index -> camera index [0..n_cameras-1]
    cam_img_indices = []
    for img_idx, img in sfm.images.items():
        if img.R is not None and img.t is not None:
            cam_img_indices.append(img_idx)

    cam_img_indices = sorted(cam_img_indices)
    imgidx_to_camidx = {img_idx: ci for ci, img_idx in enumerate(cam_img_indices)}
    n_cameras = len(cam_img_indices)
    if n_cameras == 0:
        raise RuntimeError("No cameras with poses found for bundle adjustment.")

    # (n_cameras, 6)  -> [rvec(3), t(3)]
    camera_params = np.zeros((n_cameras, 6), dtype=np.float64)
    for img_idx, cam_idx in imgidx_to_camidx.items():
        img = sfm.images[img_idx]
        R = img.R
        t = img.t.reshape(3)

        rvec, _ = cv.Rodrigues(R)
        camera_params[cam_idx, :3] = rvec.reshape(3)
        camera_params[cam_idx, 3:] = t

    valid_pt_ids = []
    for pid, pt in sfm.pts.items():
        if pt.coord is None:
            continue
        # count how many observations come from cameras that have poses
        visible_cams = 0
        for (img_idx, kp_idx) in pt.correspondences:
            if img_idx in imgidx_to_camidx:
                visible_cams += 1
            if visible_cams >= 2:
                break
        if visible_cams >= 2:
            valid_pt_ids.append(pid)

    valid_pt_ids = sorted(valid_pt_ids)
    ptid_to_local = {pid: i for i, pid in enumerate(valid_pt_ids)}
    n_points = len(valid_pt_ids)

    points_3d = np.zeros((n_points, 3), dtype=np.float64)
    for pid, local_idx in ptid_to_local.items():
        points_3d[local_idx, :] = sfm.pts[pid].coord

    # observations
    camera_indices = []
    point_indices = []
    points_2d = []

    for pid, local_idx in ptid_to_local.items():
        pt = sfm.pts[pid]
        for (img_idx, kp_idx) in pt.correspondences:
            if img_idx not in imgidx_to_camidx:
                # skip observations from images without current pose
                continue
            cam_idx = imgidx_to_camidx[img_idx]
            kp = sfm.images[img_idx].kp[kp_idx]
            u, v = kp.pt

            camera_indices.append(cam_idx)
            point_indices.append(local_idx)
            points_2d.append((u, v))

    camera_indices = np.asarray(camera_indices, dtype=np.int32)
    point_indices = np.asarray(point_indices, dtype=np.int32)
    points_2d = np.asarray(points_2d, dtype=np.float64)

    return camera_params, points_3d, camera_indices, point_indices, points_2d, imgidx_to_camidx, ptid_to_local

@profile
def runBundleAdjustment(sfm, min_points=20, verbose=1):
    """
    Run global bundle adjustment on the current SfM state.
    Updates camera poses and 3D points in-place in `sfm`.
    """
    if len(sfm.pts) < min_points:
        # Not enough structure to justify BA
        return

    (camera_params,
     points_3d,
     camera_indices,
     point_indices,
     points_2d,
     imgidx_to_camidx,
     ptid_to_local) = baBuildProblem(sfm)

    n_cameras = camera_params.shape[0]
    n_points = points_3d.shape[0]

    if n_cameras < 2 or camera_indices.size == 0:
        return

    x0 = np.hstack([camera_params.ravel(), points_3d.ravel()]).astype(np.float64)
    A = baSparsity(n_cameras, n_points, camera_indices, point_indices)

    res = least_squares(
        baResiduals,
        x0,
        jac_sparsity=A,
        verbose=verbose,
        x_scale='jac',
        ftol=1e-4,
        method='trf',
        args=(n_cameras, n_points, camera_indices, point_indices, points_2d, sfm.K),
    )

    x_opt = res.x
    cam_opt = x_opt[:n_cameras * 6].reshape((n_cameras, 6))
    pts_opt = x_opt[n_cameras * 6:].reshape((n_points, 3))

    # set camera poses back to sfm
    for img_idx, cam_idx in imgidx_to_camidx.items():
        rvec = cam_opt[cam_idx, :3].reshape(3, 1)
        tvec = cam_opt[cam_idx, 3:].reshape(3, 1)
        R_opt, _ = cv.Rodrigues(rvec)

        img = sfm.images[img_idx]
        img.setPose(R_opt.astype(np.float64), tvec.astype(np.float64))

    # set 3d points
    for pid, local_idx in ptid_to_local.items():
        sfm.pts[pid].coord = pts_opt[local_idx].astype(np.float64)

    print(f"[INFO] Bundle adjustment optimized {n_cameras} cameras, {n_points} points, "
          f"{camera_indices.size} observations. Final cost: {res.cost:.3f}")
