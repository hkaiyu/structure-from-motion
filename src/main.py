"""
Dataset references:
- https://github.com/erik-norlin/Structure-From-Motion
- https://www.cs.cornell.edu/projects/bigsfm/
- https://huggingface.co/datasets/whc/fastmap_sfm/blob/main/README.md
"""

import os
import threading
import numpy as np
import cv2 as cv
from itertools import combinations
from pathlib import Path
import argparse
import pycolmap

import dataset_loader
from colmap_loader import load_scene_colmap, ensure_colmap_model
from point_cloud_viewer import PointCloudViewer
from feature_extractor import FeatureExtractor
from feature_matcher import FeatureMatcher
from image_data import ImageData
from point_data import PointData
from bundle_adjustment import runBundleAdjustment
from evaluation import evaluateAccuracy
from utils import estimatePose, triangulate, loadAllImages, reprojectionError, pointDepth, parallaxAngle, profile

class SfmData:
    def __init__(self):
        # store dict of image data
        self.images = {}
        # store dict of point data
        self.pts = {}
        self.imageCount = 0
        self.pointCount = 0
        # camera matrix
        self.K = None
        # feature extractor/matcher
        self.extractor = None
        self.matcher = None
        # pairwise matches cache: key is (i, j) with i < j, value is list[cv.DMatch]
        self.pairMatches = {}
        self.kp_to_point = {}  # (img_idx, kp_idx) -> point_id
        # point observations stored in pt, pt.correspondences

    def setCameraIntrinsics(self, K):
        self.K = K

    # Add image from a path or preloaded ndarray, with optional precomputed features
    # If keypoints/descriptors are provided, they are used directly to avoid recomputation
    def addImage(self, image_or_path, keypoints=None, descriptors=None):
        img = ImageData()

        img.idx = self.imageCount
        self.imageCount += 1

        # Load/assign image
        if isinstance(image_or_path, (str, os.PathLike)):
            img.path = str(image_or_path)
            img.img = cv.imread(img.path)
        else:
            img.img = image_or_path
        
        # img.img_gray = cv.cvtColor(img.img, cv.COLOR_BGR2GRAY)

        # Assign or compute features
        if keypoints is not None and descriptors is not None:
            img.kp = keypoints
            img.des = descriptors
        else:
            kp, des = self.extractor.detectAndCompute(img.img)
            img.kp = kp
            img.des = des

        self.images[img.idx] = img
        return img.idx
    
    def getImage(self, img_index):
        return self.images.get(img_index, None)

    def getPointCloud(self):
        pts = []
        colors = []
        for pt in self.pts.values():
            if pt.coord is None:
                continue
            pts.append(pt.coord)

            if pt.color is None:
                colors.append([0.6, 0.8, 1.0])  # fallback color
            else:
                colors.append(pt.color)

        return (np.array(pts, np.float32), np.array(colors, np.float32))

    def getCameraData(self):
        cams = []
        for img_idx, img in self.images.items():
            if img.R is None or img.t is None:
                continue
            cams.append({
                "R": img.R,
                "t": img.t,
                "K": self.K,
                "name": f"Camera {img_idx}",
                "color": "orange"
            })
        return cams

    # add point, will add 3d point and relevant correspondences
    def addPoint(self, coord, img_i_idx, img_i_kp_idx, img_j_idx, img_j_kp_idx):
        pt = PointData()
        
        pt.idx = self.pointCount
        self.pointCount += 1
        
        pt.coord = coord
        
        # Store point index with each associated image
        pt.addCorrespondence(img_i_idx, img_i_kp_idx, sfm=self)
        pt.addCorrespondence(img_j_idx, img_j_kp_idx, sfm=self)
        
        # App sfm track of 2d keypoints to 3d point
        self.kp_to_point[(img_i_idx, img_i_kp_idx)] = pt.idx
        self.kp_to_point[(img_j_idx, img_j_kp_idx)] = pt.idx
        
        self.pts[pt.idx] = pt
        return pt.idx
    
    def getPoint(self, pt_index):
        return self.pts.get(pt_index, None)

    def setExtractor(self, extractor):
        self.extractor = extractor
                
    def setMatcher(self, matcher):
        self.matcher = matcher

    def genSIFTMatchPairs(self, img1, img2):
        if img1 is None or img2 is None:
            raise ValueError("img1 and img2 must be valid ImageData instances")

        # Use Keypoints and Descriptors already found within img1 and img2 imgData class
        kp1, des1 = img1.kp, img1.des
        kp2, des2 = img2.kp, img2.des
        if des1 is None or des2 is None:
            raise ValueError("Descriptors missing. Ensure features are extracted before matching.")

        i, j = img1.idx, img2.idx
        key = (min(i, j), max(i, j))

        # Search for corresponding pairs of points
        matches_12 = self.matcher.knnMatch(des1, des2, ratioVal=0.75)
        matches_21 = self.matcher.knnMatch(des2, des1, ratioVal=0.75)

        # Symmetric consistency check
        mutual = []
        for m in matches_12:
            for m2 in matches_21:
                if m.queryIdx == m2.trainIdx and m.trainIdx == m2.queryIdx:
                    mutual.append(m)
                    break

        if len(mutual) < 8:
            return np.empty((0,2)), np.empty((0,2)), [], kp1, kp2

        matches = sorted(mutual, key=lambda m: m.distance)

        # Here we are just trying to constrain tnat i<j
        kp_pairs = []
        if i < j:
            kp_pairs = [(m.queryIdx, m.trainIdx) for m in matches]
        else:
            kp_pairs = [(m.trainIdx, m.queryIdx) for m in matches]

        self.pairMatches[key] = {
            "matches": matches,
            "kp_pairs": kp_pairs
        }
        pts1 = np.float32([kp1[pair[0]].pt for pair in kp_pairs])
        pts2 = np.float32([kp2[pair[1]].pt for pair in kp_pairs])

        return pts1, pts2, matches, kp1, kp2

# @profile
def buildCorrespondences(sfm, new_img_idx, min_track_len=2):
    """
    Build 2D-3D correspondences for a new image using existing 3D points.
      - look at matches between image k and new image
      - if the keypoint in image k is already associated with a 3D point, we use that point as the 3D coordinate and
        the matched keypoint in the new image as the 2D observation.

    Returns:
        obj_points: (N, 3) float32
        img_points: (N, 2) float32
    """
    new_img = sfm.getImage(new_img_idx)
    obj_points = []
    img_points = []
    used_new_kps = set()

    for k, img_k in sfm.images.items():
        if k == new_img_idx or not img_k.triangulated:
            continue
        key = (min(k, new_img_idx), max(k, new_img_idx))
        pm = sfm.pairMatches.get(key)
        if not pm:
            continue

        kp_pairs = pm["kp_pairs"]
        matches = pm["matches"]

        good = [i for i,m in enumerate(matches) if m.distance < 55]
        if not good:
            continue

        for gi in good:
            kp_i, kp_j = kp_pairs[gi]
            kp_idx_tri = kp_i if k < new_img_idx else kp_j
            kp_idx_new = kp_j if k < new_img_idx else kp_i

            if kp_idx_new in used_new_kps:
                continue
            used_new_kps.add(kp_idx_new)

            pt_id = sfm.kp_to_point.get((k, kp_idx_tri))
            if pt_id is None:
                continue

            pt = sfm.getPoint(pt_id)
            obj_points.append(pt.coord)
            img_points.append(new_img.kp[kp_idx_new].pt)

    if len(obj_points) == 0:
        return np.empty((0, 3), np.float32), np.empty((0, 2), np.float32)

    return np.asarray(obj_points, np.float32), np.asarray(img_points, np.float32)

def chooseNextImage(sfm, registered, failed_images, min_correspondences=6, alpha=0.7, beta=0.3):
    best_j = None
    best_score = -1.0

    for j, img_j in sfm.images.items():
        if j in registered or j in failed_images:
            continue
        if img_j.des is None:
            continue

        obj_pts = []
        img_pts = []
        used = set()

        for k in registered:
            key = (min(k, j), max(k, j))
            pm = sfm.pairMatches.get(key)
            if not pm:
                continue

            matches = pm["matches"]
            kp_pairs = pm["kp_pairs"]

            good = [idx for idx, m in enumerate(matches) if m.distance < 55]
            if len(good) < 3:
                continue

            img_k = sfm.images[k]

            for gi in good:
                kp_k, kp_j = kp_pairs[gi]
                if k > j:
                    kp_k, kp_j = kp_j, kp_k

                if kp_j in used:
                    continue
                used.add(kp_j)

                pt_id = sfm.kp_to_point.get((k, kp_k))
                if pt_id is None:
                    continue

                pt = sfm.getPoint(pt_id)
                if pt is None or pt.coord is None:
                    continue

                obj_pts.append(pt.coord)
                img_pts.append(img_j.kp[kp_j].pt)

        obj_pts = np.asarray(obj_pts, np.float32)
        img_pts = np.asarray(img_pts, np.float32)
        M_existing = obj_pts.shape[0]

        if M_existing < min_correspondences:
            continue

        M_new = 0
        for k in registered:
            key = (min(k, j), max(k, j))
            pm = sfm.pairMatches.get(key)
            if not pm:
                continue

            kp_pairs = pm["kp_pairs"]
            for kp_k_idx, kp_j_idx in kp_pairs:
                if k > j:
                    kp_k_idx, kp_j_idx = kp_j_idx, kp_k_idx
                if (j, kp_j_idx) not in sfm.kp_to_point:
                    M_new += 1

        score = alpha * float(M_existing) + beta * float(M_new)

        if score > best_score:
            best_score = score
            best_j = j

    if best_j is None:
        print("[WARN] No suitable next image found.")
        return None

    print(f"[INFO] Next image chosen: {best_j}  score={best_score:.1f}")
    return best_j

def pickInitialPair(sfm, max_candidates=10, sample_points=200):
    """
    Choose a seed pair inside this component that provides good parallax.

    Returns:
        (i0, j0), match_count
    """
    pairs = []
    for i in sfm.images.keys():
        for j in sfm.images.keys():
            if i >= j:
                continue
            key = (min(i, j), max(i, j))
            if key in sfm.pairMatches:
                n_matches = len(sfm.pairMatches[key]["kp_pairs"])
                if n_matches >= 20:
                    pairs.append((i, j, n_matches))
    if not pairs:
        return (None, None), 0

    pairs.sort(key=lambda x: x[2], reverse=True)
    candidates = pairs[:max_candidates]

    best_score = -1
    best_pair = None
    best_match_count = 0

    for (i0, j0, n_matches) in candidates:
        key = (min(i0, j0), max(i0, j0))
        kp_pairs = sfm.pairMatches[key]["kp_pairs"]
        total = len(kp_pairs)

        if total > sample_points:
            idx = np.random.choice(total, size=sample_points, replace=False)
            sub_pairs = [kp_pairs[k] for k in idx]
        else:
            sub_pairs = kp_pairs

        if len(sub_pairs) < 8:
            continue

        pts_i = np.float32([sfm.images[i0].kp[a].pt for a, b in sub_pairs])
        pts_j = np.float32([sfm.images[j0].kp[b].pt for a, b in sub_pairs])

        E, mask_E = cv.findEssentialMat(
            pts_i, pts_j, sfm.K,
            method=cv.RANSAC,
            prob=0.999,
            threshold=1.0
        )
        if mask_E is None:
            continue

        mask_E = mask_E.ravel().astype(bool)
        pts_i = pts_i[mask_E]
        pts_j = pts_j[mask_E]

        # Need some inliers for usable geometry
        if pts_i.shape[0] < 30:
            continue

        # Triangulate relative baseline only (R1=I, t1=[0])
        _, R, t, _ = cv.recoverPose(E, pts_i, pts_j, sfm.K)
        pts3d = triangulate(
            pts_i, pts_j,
            np.eye(3), np.zeros((3, 1)),
            R, t,
            sfm.K,
            decimal_pts=2
        )

        # Skip if degenerate triangulation
        if pts3d.shape[0] < 10:
            continue

        # Compute SVD of centered structure
        Xc = pts3d - pts3d.mean(axis=0)
        U, S, Vt = np.linalg.svd(Xc)

        if S[0] <= 0:
            continue

        nonplanarity = S[2] / S[0]
        if nonplanarity > best_score:
            best_score = nonplanarity
            best_pair = (i0, j0)
            best_match_count = n_matches

    if best_pair is None:
        i0, j0, best_match_count = pairs[0]
        print("[WARN] Using fallback seed pair.")
        return (i0, j0), best_match_count

    print(f"[INFO] Selected seed pair {best_pair} with parallax score {best_score:.4f}")
    return best_pair, best_match_count

@profile
def triangulateWithExistingCameras(sfm, new_idx, reproj_thresh=2.0):
    new_img = sfm.getImage(new_idx)

    for k, img_k in sfm.images.items():
        if k == new_idx or not img_k.triangulated:
            continue

        key = (min(k, new_idx), max(k, new_idx))
        pm = sfm.pairMatches.get(key)
        if not pm:
            continue

        matches = pm["matches"]
        kp_pairs = pm["kp_pairs"]

        # Basic descriptor-quality filter
        good = [i for i, m in enumerate(matches) if m.distance < 55.0]
        if len(good) < 8:
            continue

        # Build raw 2D–2D arrays with correct index mapping
        pts_k = []
        pts_new = []
        idx_k = []
        idx_new = []

        for gi in good:
            kp_i, kp_j = kp_pairs[gi]

            if k < new_idx:
                # keypoint order in kp_pairs: (k, new_idx)
                idx_k.append(kp_i)
                idx_new.append(kp_j)
                pts_k.append(img_k.kp[kp_i].pt)
                pts_new.append(new_img.kp[kp_j].pt)
            else:
                # if ever used with reversed order, keep it consistent
                idx_k.append(kp_j)
                idx_new.append(kp_i)
                pts_k.append(img_k.kp[kp_j].pt)
                pts_new.append(new_img.kp[kp_i].pt)

        pts_k = np.float32(pts_k)
        pts_new = np.float32(pts_new)
        idx_k = np.asarray(idx_k)
        idx_new = np.asarray(idx_new)

        if pts_k.shape[0] < 8:
            continue

        # filter with essential mat
        E, mask_E = cv.findEssentialMat(
            pts_k, pts_new, sfm.K,
            method=cv.RANSAC, prob=0.999, threshold=1.0
        )
        if mask_E is None:
            continue

        mask_E = mask_E.ravel().astype(bool)
        if mask_E.sum() < 8:
            continue

        pts_k_f = pts_k[mask_E]
        pts_new_f = pts_new[mask_E]
        idx_k_f = idx_k[mask_E]
        idx_new_f = idx_new[mask_E]

        # triangulate
        pts3d = triangulate(
            pts_k_f, pts_new_f,
            img_k.R, img_k.t,
            new_img.R, new_img.t,
            sfm.K,
            decimal_pts=2
        )

        # Per-point cheirality / parallax / reprojection checks
        for X, kp_k, kp_new, u_k, u_new in zip(pts3d, idx_k_f, idx_new_f, pts_k_f, pts_new_f):
            # cheirality
            depth_k = pointDepth(img_k.R, img_k.t, X)
            depth_new = pointDepth(new_img.R, new_img.t, X)
            if depth_k <= 0 or depth_new <= 0:
                continue

            # parallax
            angle = parallaxAngle(img_k.R, img_k.t, new_img.R, new_img.t, X)
            if angle < 1.0:   # in degrees; you can tune to 0.5 or 1.5 if needed
                continue

            # reprojection
            if (reprojectionError(sfm.K, img_k.R, img_k.t, X, u_k) > reproj_thresh or
                reprojectionError(sfm.K, new_img.R, new_img.t, X, u_new) > reproj_thresh):
                continue

            # Merge with existing point if any
            pt_id = (sfm.kp_to_point.get((k, kp_k)) or
                     sfm.kp_to_point.get((new_idx, kp_new)))
            if pt_id is not None:
                pt = sfm.getPoint(pt_id)
                if pt is not None:
                    pt.addCorrespondence(new_idx, kp_new, sfm)
                continue

            sfm.addPoint(X, k, kp_k, new_idx, kp_new)

def getIntrinsics(model, params):
    """
    We don't account for distortion in our pipeline, so if "RADIAL" or "SIMPLE_RADIAL" is detected,
    we will just try treating it as pinhole.
    """
    if model == "SIMPLE_PINHOLE":
        f, cx, cy = params
        K = np.array([
            [f, 0, cx],
            [0, f, cy],
            [0, 0, 1]
        ])

    elif model == "PINHOLE":
        fx, fy, cx, cy = params
        K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])

    elif model == "SIMPLE_RADIAL":
        f, cx, cy, k1 = params
        K = np.array([
            [f, 0, cx],
            [0, f, cy],
            [0, 0, 1]
        ])

    elif model == "RADIAL":
        f, cx, cy, k1, k2 = params
        K = np.array([
            [f, 0, cx],
            [0, f, cy],
            [0, 0, 1]
        ])
    else:
        raise NotImplementedError(f"Camera model {model} not handled.")
    return K

@profile
def runColmap(images_dir, workspace):
    os.makedirs(workspace, exist_ok=True)

    database_path = os.path.join(workspace, "database.db")
    sparse_dir = os.path.join(workspace, "sparse")
    os.makedirs(sparse_dir, exist_ok=True)

    pycolmap.extract_features(
        database_path=database_path,
        image_path=images_dir,
    )

    pycolmap.match_exhaustive(
        database_path=database_path,
    )

    maps = pycolmap.incremental_mapping(
        database_path=database_path,
        image_path=images_dir,
        output_path=sparse_dir,
    )
    if maps:
        maps[0].write(sparse_dir)

    return pycolmap.Reconstruction(sparse_dir)

@profile
def sfmRun(dataset, viewer):
    """
    Assumptions:
        - Calibrated SfM (no fundamental matrix needed, directly calculate essential matrix with intrinsics)
        - Incremental (not global SfM)
        - Single camera per dataset
    """
    colmap = runColmap(dataset.image_dir, dataset.colmap_dir)
    if dataset.K is None:
        cam = next(iter(colmap.cameras.values()))
        dataset.K = getIntrinsics(cam.model, cam.params)

    sfm = SfmData()
    sfm.setExtractor(FeatureExtractor("SIFT", nfeatures=10000, contrastThreshold=0.03))
    sfm.setMatcher(matcher = FeatureMatcher("BF", norm=cv.NORM_L2, crossCheck=False))
    sfm.setCameraIntrinsics(dataset.K)

    # ==========================================================
    # 1. Load images and extract features
    # ==========================================================
    print(f"[INFO] Loading in images under {dataset.image_dir}")
    images = loadAllImages(dataset.image_dir)
    features = {}
    img_indices = []
    for img_id, imgdata in enumerate(images):
        fp, img = imgdata 
        # store original path for evaluation/reporting
        try:
            idx = sfm.addImage(img)
            sfm.images[idx].path = fp
            img_indices.append(idx)
        except Exception as e:
            print(e)
    print(f"[INFO] Loaded {len(img_indices)} images")

    # ==========================================================
    # 2. Match features between image pairs
    # ==========================================================
    # compute pairwise matches for pairs within a window (much faster on larger datasets compared to all pairs)
    # NOTE: we take advantage of the fact that images close to each other in each dataset are typically close in the
    # scene, so this would not be good if a dataset were to have images randomly shuffled
    window = 8
    for i in range(len(images)):
        for j in range(i+1, min(i+1+window, len(images))):
            img_i = sfm.getImage(i)
            img_j = sfm.getImage(j)
            if img_i.des is None or img_j.des is None:
                print(f"[WARN] Skipping pair ({i},{j}) due to missing descriptors")
                continue
            pts1, pts2, pair_matches, kp1, kp2 = sfm.genSIFTMatchPairs(img_i, img_j)
            print(f"Matched pair ({i}, {j}): {len(pair_matches)} matches")

    # ==========================================================
    # 3. Iterative triangulation and addition to point cloud
    # ==========================================================    
    (i0, j0), n_matches = pickInitialPair(sfm)
    if (i0, j0) is None:
        raise RuntimeError("Cannot find a valid initialization pair.")

    print(f"[INFO] Global seed pair ({i0}, {j0}) with {n_matches} matches")

    img_i = sfm.getImage(i0)
    img_j = sfm.getImage(j0)

    img_i.setPose(np.eye(3, dtype=np.float32), np.zeros((3, 1), dtype=np.float32))
    img_i.triangulated = True

    # Load symmetric matches for (i0, j0)
    key = (min(i0, j0), max(i0, j0))
    entry = sfm.pairMatches[key]
    best_matches = entry["kp_pairs"]

    pts_i = np.float32([sfm.images[i0].kp[kp_i].pt for kp_i, kp_j in best_matches])
    pts_j = np.float32([sfm.images[j0].kp[kp_j].pt for kp_i, kp_j in best_matches])

    E, mask_E = cv.findEssentialMat(
        pts_i, pts_j, sfm.K,
        method=cv.RANSAC,
        prob=0.999,
        threshold=1.0
    )
    mask_E = mask_E.ravel().astype(bool)

    pts_i = pts_i[mask_E]
    pts_j = pts_j[mask_E]
    inlier_pairs = [p for p, m in zip(best_matches, mask_E) if m]

    R_ij, t_ij, _ = estimatePose(pts_i, pts_j, E, sfm.K)

    img_j.setPose(R_ij, t_ij)
    img_j.triangulated = True

    pt_cloud = triangulate(pts_i, pts_j,
                           img_i.R, img_i.t,
                           img_j.R, img_j.t,
                           sfm.K)

    for (X, (kp_i, kp_j)) in zip(pt_cloud, inlier_pairs):
        sfm.addPoint(X, i0, kp_i, j0, kp_j)

    viewer.updateCameraPoses(sfm.getCameraData())
    viewer.updatePoints(*sfm.getPointCloud())

    registered = {i0, j0}
    failed_images = set()
    triangulatedCount = 2

    while True:
        next_idx = chooseNextImage(sfm, registered, failed_images)
        if next_idx is None:
            print("[INFO] No more images to register.")
            break

        img_j = sfm.getImage(next_idx)

        obj_points, img_points = buildCorrespondences(sfm, next_idx)
        if obj_points.shape[0] < 6:
            print(f"[WARN] Not enough 2D–3D matches for image {next_idx}")
            failed_images.add(next_idx)
            continue

        success, rvec, tvec, inliers_pnp = cv.solvePnPRansac(
            obj_points,
            img_points,
            sfm.K,
            None,
            flags=cv.SOLVEPNP_ITERATIVE,
            iterationsCount=200,
            reprojectionError=4.0,
            confidence=0.999
        )

        if (not success) or inliers_pnp is None or len(inliers_pnp) < 6:
            print(f"[WARN] PnP failed for image {next_idx}")
            failed_images.add(next_idx)
            continue

        R_j, _ = cv.Rodrigues(rvec)
        img_j.setPose(R_j.astype(np.float32), tvec.reshape(3, 1).astype(np.float32))
        img_j.triangulated = True
        registered.add(next_idx)
        triangulatedCount += 1

        triangulateWithExistingCameras(sfm, next_idx)
        if triangulatedCount >= 4 and triangulatedCount % 3 == 0:
            runBundleAdjustment(sfm, min_points=50, verbose=0)

        viewer.updatePoints(*sfm.getPointCloud())
        viewer.updateCameraPoses(sfm.getCameraData())

    print("[INFO] Point cloud construction completed.")

    print("\n[INFO] Running final evaluation vs GT...")
    metrics = evaluateAccuracy(sfm, colmap)
    print(" POSE ERRORS:", metrics["pose_errors"])
    print(" POINT CLOUD ERRORS:", metrics["pc_errors"])

    # For debug, I would reccommend just printing terminal output to a file if you use this
    # for pt_id, pt in sfm.pts.items():
    #     print(f"Point {pt_id}: 3D coords = {pt.coord}")
    #     for img_idx, kp_idx in pt.correspondences:
    #         print(f"  seen in image {img_idx}, keypoint {kp_idx}")
    # for (img_idx, kp_idx), pt_id in sfm.kp_to_point.items():
    #     print(f"Image {img_idx}, keypoint {kp_idx} -> Point {pt_id}, 3D = {sfm.pts[pt_id].coord}")

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=None, help="(optional) path to dataset directory)")
    args = parser.parse_args()

    dataset_path = Path(args.dataset).resolve() if args.dataset else None
    if (not dataset_path) or (not dataset_path.is_dir()):
        print("No or invalid dataset path provided. Entering CLI dataset selection")
        dataset_dir = Path(project_root / "dataset")
        dataset = dataset_loader.select_dataset(dataset_dir)
    else:
        dataset = dataset_loader.DatasetInfo(dataset_path, focal_length=False, from_cli=False)

    viewer = PointCloudViewer("SfM Point Cloud")
    threading.Thread(target=sfmRun, args=(dataset, viewer), daemon=True).start()
    viewer.run()
