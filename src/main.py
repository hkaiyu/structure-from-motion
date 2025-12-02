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

import dataset_loader
from colmap_loader import load_scene_colmap, ensure_colmap_model
from eth3d_eval import evaluatePointCloud
from point_cloud_viewer import PointCloudViewer
from feature_extractor import FeatureExtractor
from feature_matcher import FeatureMatcher
from image_data import ImageData
from point_data import PointData
from bundle_adjustment import runBundleAdjustment
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

def chooseNextImage(sfm, current_idx, img_indices, failed_images):
    best_j = None
    best_inliers = 0
    img_i = sfm.getImage(current_idx)

    for j in img_indices:
        if j == current_idx:
            continue

        img_j = sfm.getImage(j)
        if j in failed_images or img_j.triangulated or img_i.des is None or img_j.des is None:
            continue

        key = (min(current_idx, j), max(current_idx, j))
        pm = sfm.pairMatches.get(key)
        if not pm:
            continue

        matches = pm["matches"]
        kp_pairs = pm["kp_pairs"]

        good = [idx for idx, m in enumerate(matches) if m.distance < 55]
        if len(good) < 8:
            continue

        pts_i = np.float32([img_i.kp[kp_pairs[g][0]].pt for g in good])
        pts_j = np.float32([img_j.kp[kp_pairs[g][1]].pt for g in good])

        E, mask_E = cv.findEssentialMat(
            pts_i, pts_j, sfm.K,
            method=cv.RANSAC, prob=0.999, threshold=1.0
        )
        if mask_E is None:
            continue

        inliers = int(mask_E.sum())
        if inliers > best_inliers:
            best_inliers = inliers
            best_j = j

    if best_j is None or best_inliers < 8:
        print(f"[WARN] No good next image found from {current_idx}. best_inliers={best_inliers}")
        return None

    print(f"[INFO] Next image: {best_j} with {best_inliers} inliers")
    return best_j

def buildViewGraph(sfm, min_pair_overlap=20):
    """
    view_graph[i][j] = number of mutual matches between image i and j
    """
    view_graph = {i: {} for i in range(sfm.imageCount)}
    for (i, j), entry in sfm.pairMatches.items():
        n = len(entry["kp_pairs"])
        if n >= min_pair_overlap:
            view_graph[i][j] = n
            view_graph[j][i] = n
    return view_graph


def findComponents(sfm, view_graph):
    """
    Find connected components in the view graph via standard graph traversal.
    Returns: list of components, each a list of image indices.
    """
    visited = set()
    components = []

    for v in range(sfm.imageCount):
        if v not in visited:
            stack = [v]
            comp = []
            while stack:
                u = stack.pop()
                if u in visited:
                    continue
                visited.add(u)
                comp.append(u)
                for nbr in view_graph[u].keys():
                    if nbr not in visited:
                        stack.append(nbr)
            if len(comp) > 1:
                components.append(sorted(comp))

    return components

def chooseNextImageComponent(sfm, registered, comp, failed_images,
                                view_graph=None,
                                alpha=0.7, beta=0.3,
                                min_2d3d=6):
    """
    Choose the next image to register inside a given component.

    - `registered`: set of already triangulated image indices
    - `comp`: list of image indices in this component
    - `failed_images`: images we already tried and failed PnP on
    - `view_graph`: optional adjacency dict
    """
    best_j = None
    best_score = -1.0

    for j in comp:
        if j in registered or j in failed_images:
            continue

        img_j = sfm.getImage(j)
        if img_j is None or img_j.des is None:
            continue

        # 1) Collect 2D–3D correspondences for this candidate j
        obj_points = []
        img_points = []
        used_kp_j = set()

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

            img_k = sfm.getImage(k)
            for gi in good:
                kp_k, kp_j = kp_pairs[gi]
                if k > j:
                    kp_k, kp_j = kp_j, kp_k

                if kp_j in used_kp_j:
                    continue
                used_kp_j.add(kp_j)

                pt_id = sfm.kp_to_point.get((k, kp_k))
                if pt_id is None:
                    continue
                pt = sfm.getPoint(pt_id)
                if pt is None or pt.coord is None:
                    continue

                obj_points.append(pt.coord)
                img_points.append(img_j.kp[kp_j].pt)

        obj_points = np.asarray(obj_points, np.float32)
        img_points = np.asarray(img_points, np.float32)
        M_existing = obj_points.shape[0]

        # If we have almost no 2D–3D support yet, skip this candidate for now.
        if M_existing < min_2d3d:
            continue

        # 2) Count potential new structure: 2D matches that don't yet map to 3D points
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
                if (j, kp_j_idx) not in sfm.kp_to_point: # new observation
                    M_new += 1

        # 3) View-graph degree bonus
        degree = len(view_graph.get(j, []))
        gamma = 0.05

        # so we take into a mix of observations existing between w/ observations that would add new structure
        # NOTE: this was an attempt to try to get introduce more structure and not fall flat onto dominant plane...
        #       as it stands, this is not sufficient
        score = alpha * float(M_existing) + beta * float(M_new) + gamma * float(degree)

        if score > best_score:
            best_score = score
            best_j = j

    if best_j is None:
        print("[WARN] No suitable next image found in this component.")
        return None

    print(f"[INFO] Next image (component-wise): {best_j}, score={best_score:.2f}")
    return best_j

def estimateSimilarityRansac(src, dst, num_iters=1000, sample_size=3, threshold=0.05, min_inliers=15):
    """
    Robust similarity (scale, R, t) from src -> dst using RANSAC on 3D-3D correspondences.
    src, dst: (N,3)
    Returns: (scale, R, t, inlier_mask) or (None, None, None, None) if failed.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    assert src.shape == dst.shape
    N = src.shape[0]
    if N < sample_size:
        return None, None, None, None

    best_inliers = None
    best_inlier_count = 0

    rng = np.random.default_rng()

    for _ in range(num_iters):
        idx = rng.choice(N, size=sample_size, replace=False)
        s_hat, R_hat, t_hat = estimate_similarity_umeyama(src[idx], dst[idx])

        X_src = src
        X_dst_pred = (s_hat * (R_hat @ X_src.T) + t_hat).T
        err = np.linalg.norm(X_dst_pred - dst, axis=1)
        inliers = err < threshold
        count = int(inliers.sum())
        if count > best_inlier_count:
            best_inlier_count = count
            best_inliers = inliers

    if best_inliers is None or best_inlier_count < min_inliers:
        return None, None, None, None

    # Re-estimate using all inliers
    s_final, R_final, t_final = estimate_similarity_umeyama(src[best_inliers], dst[best_inliers])
    return s_final, R_final, t_final, best_inliers

def mergeComponents(sfm, components, view_graph):
    """
    Merge independently reconstructed components into the frame of the first component.
    This is necessary when we essentially reconstruct multiple "islands" of point clouds and need something
    to relate them geometrically back together
    """

    if len(components) <= 1:
        print("[INFO] Only one component; no merging needed.")
        return

    base_comp = components[0]
    base_set = set(base_comp)

    for k in range(1, len(components)):
        target_comp = components[k]
        target_set = set(target_comp)
        print(f"\n[INFO] Merging component {k} into base frame...")

        pts_base = []
        pts_target = []

        # ------------------------------------------------------------------
        # 1. Collect 3D-3D correspondences from 2D matches between components
        # ------------------------------------------------------------------
        for i in base_set:
            for j in target_set:
                key = (min(i, j), max(i, j))
                pm = sfm.pairMatches.get(key)
                if not pm:
                    continue

                kp_pairs = pm["kp_pairs"]
                for kp_i, kp_j in kp_pairs:
                    pA = sfm.kp_to_point.get((i, kp_i))
                    pB = sfm.kp_to_point.get((j, kp_j))
                    if pA is None or pB is None:
                        continue
                    ptA = sfm.getPoint(pA)
                    ptB = sfm.getPoint(pB)
                    if ptA is None or ptB is None:
                        continue
                    if ptA.coord is None or ptB.coord is None:
                        continue

                    pts_base.append(ptA.coord)
                    pts_target.append(ptB.coord)

        pts_base = np.asarray(pts_base, dtype=np.float64)
        pts_target = np.asarray(pts_target, dtype=np.float64)

        print(f"[INFO] Component {k}: found {len(pts_base)} raw 3D-3D pairs.")

        if pts_base.shape[0] < 8:
            print("[WARN] Not enough 3D-3D correspondences to merge this component; skipping.")
            continue

        # ------------------------------------------------------------------
        # 2. RANSAC similarity from target -> base
        # ------------------------------------------------------------------
        scale, R_s, t_s, inliers = estimateSimilarityRansac(
            src=pts_target,
            dst=pts_base,
            num_iters=1000,
            sample_size=3,
            threshold=0.05,   # tune depending on scene scale
            min_inliers=15
        )

        if scale is None:
            print("[WARN] Failed to robustly estimate similarity for component", k)
            continue

        print(f"[INFO] Component {k}: similarity estimated with {inliers.sum()} inliers. scale={scale:.4f}")

        # ------------------------------------------------------------------
        # 3. Apply similarity to cameras and points of target component
        # ------------------------------------------------------------------
        target_image_set = set(target_comp)

        # Cameras
        for idx in target_comp:
            img = sfm.getImage(idx)
            if img is None or img.R is None or img.t is None:
                continue

            R_cam = img.R.astype(np.float64)
            t_cam = img.t.astype(np.float64)  # (3,1) or (3,)

            if t_cam.ndim == 1:
                t_cam = t_cam.reshape(3, 1)

            R_new = R_cam @ R_s.T
            t_new = scale * t_cam - R_cam @ R_s.T @ t_s

            img.R = R_new.astype(np.float32)
            img.t = t_new.astype(np.float32)

        # Points
        for pidx, pt in sfm.pts.items():
            if pt.coord is None:
                continue
            # If this point belongs only to target images, transform it
            if any((img_idx in target_image_set) for (img_idx, _kp_idx) in pt.correspondences):
                X = pt.coord.astype(np.float64).reshape(3, 1)
                X_new = scale * (R_s @ X) + t_s
                pt.coord = X_new.flatten().astype(np.float32)

        print(f"[INFO] Component {k}: transformed into base coordinate frame.")

    print("[INFO] All components that could be merged are now in the same frame.")

def pickInitialPair(sfm, comp, max_candidates=10, sample_points=200):
    """
    Choose a seed pair inside this component that provides good parallax.

    Returns:
        (i0, j0), match_count
    """

    # Collect candidate matches inside component
    pairs = []
    for i in comp:
        for j in comp:
            if i >= j:
                continue
            key = (min(i, j), max(i, j))
            if key in sfm.pairMatches:
                n_matches = len(sfm.pairMatches[key]["kp_pairs"])
                if n_matches >= 20:
                    pairs.append((i, j, n_matches))

    if not pairs:
        return (None, None), 0

    # Sort by match count (descending) and keep top K
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

    # Fallback if no candidate is geometrically good
    if best_pair is None:
        i0, j0, best_match_count = pairs[0]
        print("[WARN] Using fallback seed pair on match count only.")
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
@profile
def sfmRun(dataset, viewer):
    """
    Assumptions:
        - Calibrated SfM (no fundamental matrix needed, directly calculate essential matrix with intrinsics)
        - Incremental (not global SfM)
        - Single camera per dataset
    """
    dataset_dir = dataset.dataset_dir
    print(f"[INFO] Loading dataset: {dataset_dir}")

    # Ensure COLMAP TXT model exists for this dataset (or run COLMAP if missing)
    try:
        ensure_colmap_model(Path(dataset_dir))
    except Exception as e:
        print(f"[WARN] ensure_colmap_model encountered an error: {e}")

    images = None
    intrinsics = None
    extrinsics = None
    gt = None

    try:
        images, intrinsics, extrinsics, gt = load_scene_colmap(dataset_dir)
        if len(images) > 0 and intrinsics is not None:
            print("[INFO] Detected COLMAP model.")
        else:
            raise ValueError("COLMAP load returned empty or invalid data")

    except Exception as e:
        print(f"[WARN] COLMAP load failed: {e}")
        print("[INFO] Falling back to generic image folder loader. Accuracy not guaranteed!")

        # Fallback: treat dataset_dir as a directory of images only
        # We assume no ground truth in this case
        img_files = sorted(list(Path(dataset_dir).glob("*.[jp][pn]g")))
        if len(img_files) == 0:
            raise FileNotFoundError(f"No images found in {dataset_dir}")

        images = img_files
        intrinsics = None
        extrinsics = None
        gt = None

    sfm = SfmData()
    sfm.setExtractor(FeatureExtractor("SIFT", nfeatures=10000, contrastThreshold=0.03))
    # sfm.setMatcher(matcher = FeatureMatcher("FLANN"))
    # sfm.setExtractor(FeatureExtractor("SIFT", nfeatures=5000))
    sfm.setMatcher(matcher = FeatureMatcher("BF", norm=cv.NORM_L2, crossCheck=False))

    # ==========================================================
    # 1. Load images and extract features
    # ==========================================================
    print("[INFO] Loading in images...")
    image_paths = list(images)
    images = loadAllImages(images)
    features = {}
    img_indices = []
    for img_id, img in enumerate(images):
        idx = sfm.addImage(img)
        # store original path for evaluation/reporting
        try:
            sfm.images[idx].path = str(image_paths[img_id])
        except Exception:
            pass
        img_indices.append(idx)
    print(f"[INFO] Loaded {len(img_indices)} images")

    if intrinsics is None:
        if dataset.from_cli:
            print("[INFO] Using intinsics stored for CLI dataset")
            H = dataset.im_height
            W = dataset.im_width
            f = dataset.focal_length
            K = dataset.K
        else:
            print("[INFO] No intrinsics found. Using approximate K")
            H, W = images[0].shape[:2]
            f = max(H, W) * 1.15 # TODO: not ideal, but we have to use some metric or do auto-calibration...
            K = np.array([[f, 0, W/2],
                        [0, f, H/2],
                        [0, 0, 1]], dtype=np.float32)
        sfm.setCameraIntrinsics(K)
    else:
        cam_ids = sorted(intrinsics.keys())
        if len(cam_ids) > 1:
            print(f"[WARN] Multiple camera intrinsics found ({cam_ids}); using camera_id={cam_ids[0]} as shared K for all images.")
        sfm.setCameraIntrinsics(intrinsics[cam_ids[0]])

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

    # Compute and cache pairwise matches for all unique pairs
    # for i, j in combinations(img_indices, 2):
    #     img_i = sfm.getImage(i)
    #     img_j = sfm.getImage(j)
    #     if img_i.des is None or img_j.des is None:
    #         print(f"[WARN] Skipping pair ({i},{j}) due to missing descriptors")
    #         continue
    #     pts1, pts2, pair_matches, kp1, kp2 = sfm.genSIFTMatchPairs(img_i, img_j)
    #     print(f"Matched pair ({i}, {j}): {len(pair_matches)} matches")

    # build view gfraph
    view_graph = buildViewGraph(sfm, min_pair_overlap=20)
    components = findComponents(sfm, view_graph)
    if not components:
        # fall back to a single component containing all images
        components = [img_indices]
    print("[INFO] View-graph components:")
    for idx, comp in enumerate(components):
        print(f"  Component {idx}: {len(comp)} images")

    # ==========================================================
    # 3. Iterative triangulation and addition to point cloud
    # ==========================================================    
    total_triangulated = 0
    for comp_idx, comp in enumerate(components):
        print(f"\n[INFO] ---- Starting component {comp_idx} ----")

        (i0, j0), n_matches = pickInitialPair(sfm, comp)
        if (i0, j0) is None:
            print(f"[WARN] No valid seed pair for component {comp_idx}, skipping.")
            continue

        print(f"[INFO] Component {comp_idx}: seed pair ({i0}, {j0}) with {n_matches} matches")

        img_i = sfm.getImage(i0)
        img_j = sfm.getImage(j0)

        if img_i.R is None or img_i.t is None:
            img_i.setPose(np.eye(3, dtype=np.float32), np.zeros((3, 1), dtype=np.float32))
        img_i.triangulated = True

        # Use the already stored matches for (i0, j0)
        key = (min(i0, j0), max(i0, j0))
        entry = sfm.pairMatches[key]
        best_matches = entry["kp_pairs"]

        pts_i = np.float32([sfm.images[i0].kp[kp_i].pt for kp_i, kp_j in best_matches])
        pts_j = np.float32([sfm.images[j0].kp[kp_j].pt for kp_i, kp_j in best_matches])

        E, mask_E = cv.findEssentialMat(pts_i, pts_j, sfm.K, cv.RANSAC, 0.999, 1.0)
        mask_E = mask_E.ravel().astype(bool)

        pts_i = pts_i[mask_E]
        pts_j = pts_j[mask_E]
        inlier_pairs = [p for p, use in zip(best_matches, mask_E) if use]

        R, t, _ = estimatePose(pts_i, pts_j, E, sfm.K)
        img_j.setPose(R, t)
        img_j.triangulated = True

        # Triangulate initial baseline points
        ptCloud = triangulate(pts_i, pts_j, img_i.R, img_i.t, img_j.R, img_j.t, sfm.K)
        for (X, (kp_i, kp_j)) in zip(ptCloud, inlier_pairs):
            sfm.addPoint(X, i0, kp_i, j0, kp_j)

        # Update viewer
        viewer.updateCameraPoses(sfm.getCameraData())
        pts_xyz, colors = sfm.getPointCloud()
        viewer.updatePoints(pts_xyz, colors)

        # Within component, do the typical incremental stuff
        registered = set([i0, j0])
        failed_images = set()
        triangulatedCount = len(registered)
        total_triangulated += (2 if total_triangulated == 0 else 0)

        while True:
            # pick next image inside this component using 2D-3D + view-graph
            next_idx = chooseNextImageComponent(
                sfm,
                registered=registered,
                comp=comp,
                failed_images=failed_images,
                view_graph=view_graph,
                alpha=0.7,
                beta=0.3,
                min_2d3d=6
            )
            if next_idx is None:
                break

            img_j = sfm.getImage(next_idx)

            # Build 2D-3D correspondences
            obj_points, img_points = buildCorrespondences(sfm, img_j.idx)
            if obj_points.shape[0] < 6:
                print(f"[WARN] Not enough 2D-3D matches for image {img_j.idx}")
                failed_images.add(img_j.idx)
                continue

            # Run PnP w/ RANSAC
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

            if (not success or inliers_pnp is None or len(inliers_pnp) < 6):
                print(f"[WARN] PnP failed or too few inliers for image {img_j.idx}")
                failed_images.add(img_j.idx)
                continue

            R_j, _ = cv.Rodrigues(rvec)
            img_j.setPose(R_j, tvec.reshape(3, 1))
            img_j.triangulated = True
            registered.add(img_j.idx)
            triangulatedCount += 1
            total_triangulated += 1

            # Triangulate new points with existing cameras
            triangulateWithExistingCameras(sfm, img_j.idx)

            # Local BA every few images (same logic as before)
            if triangulatedCount >= 4 and triangulatedCount % 3 == 0:
                runBundleAdjustment(sfm, min_points=50, verbose=0)

            # Voxel downsample + update viewer
            # voxelDownsampleFilter(sfm)
            viewer.updatePoints(*sfm.getPointCloud())
            viewer.updateCameraPoses(sfm.getCameraData())

        print(f"[INFO] Component {comp_idx} finished with {triangulatedCount} cameras registered.")
 
    # TODO: Save reconstruction to disk (e.g., PLY for points + JSON for camera poses).
    mergeComponents(sfm, components, view_graph)
    if (triangulatedCount > 5):
        print("\n[INFO] Running final global bundle adjustment...")
        runBundleAdjustment(sfm, min_points=50, verbose=2)
        # mergeClosePoints(sfm)
        viewer.updatePoints(*sfm.getPointCloud())
        viewer.updateCameraPoses(sfm.getCameraData())

    print("[INFO] Point cloud construction completed.")

    if gt is not None:
        print("\n[INFO] Running final evaluation vs GT...")
        pred_xyz, _ = sfm.getPointCloud()

        # Build predicted camera dict keyed by image index
        pred_cams = {}
        for idx, img in sfm.images.items():
            if img.R is None or img.t is None:
                continue
            pred_cams[idx] = (img.R.astype(np.float64), img.t.astype(np.float64))

        # Build GT camera dict keyed by image index (match by basename)
        gt_cams = {}
        if extrinsics is not None:
            # Pre-index GT extrinsics by basename (case-insensitive)
            ex_by_base = {}
            for name, (R, t, _cid) in extrinsics.items():
                base = Path(name).name.lower()
                ex_by_base[base] = (R, t)
            for idx in sorted(sfm.images.keys()):
                img = sfm.getImage(idx)
                if img is None or getattr(img, "path", None) is None:
                    continue
                base = Path(img.path).name.lower()
                if base in ex_by_base:
                    Rg, tg = ex_by_base[base]
                    gt_cams[idx] = (Rg.astype(np.float64), tg.astype(np.float64))

        results = evaluatePointCloud(pred_xyz, gt, pred_cams=pred_cams, gt_cams=gt_cams)

        print("\n===== Evaluation Results =====")
        for k, v in results.items():
            print(f"{k:22s}: {v}")

        # Detailed per-camera errors (if available)
        per_cam = results.get("per_camera_errors")
        if per_cam:
            print("\nPer-camera pose errors:")
            print(f"{'cam_id':>8} {'trans_err(m)':>14} {'rot_err(deg)':>14}")
            for e in sorted(per_cam, key=lambda x: x["id"]):
                print(f"{str(e['id']):>8} {e['trans_err']:14.4f} {e['rot_err_deg']:14.3f}")

            trans = np.array([e["trans_err"] for e in per_cam], dtype=float)
            rot = np.array([e["rot_err_deg"] for e in per_cam], dtype=float)

            def pct(x, p):
                return float(np.percentile(x, p)) if x.size > 0 else float("nan")

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
        

    # TODO: smaller representative set (3-4 datasets) that we allow users to pick from, we don't want repo to get too
    # big. We want the choices to include a eth3d dataset (no more since they are very large).
    
    # dataset = dataset_loader.DatasetInfo(Path(project_root / "data" / "eth3d" / "courtyard" / "images" / "dslr_images_undistorted"), focal_length=False, from_cli=False)
    # dataset = dataset_loader.DatasetInfo( Path(project_root / "dataset" / "erik" / "erik_3"), focal_length=False, from_cli=False)
    viewer = PointCloudViewer("SfM Point Cloud")
    threading.Thread(target=sfmRun, args=(dataset, viewer), daemon=True).start()
    viewer.run()
