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
from colmap import getIntrinsics, runColmap
from colmap_loader import load_scene_colmap, ensure_colmap_model
from point_cloud_viewer import PointCloudViewer
from feature_extractor import FeatureExtractor
from feature_matcher import FeatureMatcher
from image_data import ImageData
from point_data import PointData
from bundle_adjustment import runBundleAdjustment
from evaluation import evaluateAccuracy
from utils import triangulate, triangulatePoint, loadAllImages, reprojectionError, pointDepth, parallaxAngle, profile

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
        i = img1.idx
        j = img2.idx
        key = (min(i, j), max(i, j))

        full, tentative, geo_inliers = self.matcher.match(img1, img2, self.K)

        def keypointPairs(matches, swap):
            if not swap:
                return [(m.queryIdx, m.trainIdx) for m in matches]
            else:
                return [(m.trainIdx, m.queryIdx) for m in matches]

        swap = (j < i)  # always ensure (i < j)
        kp_pairs_full = keypointPairs(full, swap)
        kp_pairs_tentative = keypointPairs(tentative, swap)
        kp_pairs_inliers = keypointPairs(geo_inliers, swap)

        self.pairMatches[key] = {
            "full": full,
            "tentative": tentative,
            "inliers": geo_inliers,
            "kp_pairs_full": kp_pairs_full,
            "kp_pairs_tentative": kp_pairs_tentative,
            "kp_pairs_inliers": kp_pairs_inliers,
        }
        return self.pairMatches[key]

    def pruneObservationsForNewCamera(self, img_idx, reproj_thresh=2.0):
        img = self.images[img_idx]
        bad_point_ids = []

        for pid, pt in list(self.pts.items()):
            X = pt.coord
            to_remove = []

            for obs in pt.correspondences:
                im, kp = obs
                if im != img_idx:
                    continue

                u = img.kp[kp].pt
                err = reprojectionError(self.K, img.R, img.t, X, u)
                if err > reproj_thresh:
                    to_remove.append(obs)

            # Remove bad obs
            for obs in to_remove:
                im, kp = obs
                pt.correspondences.remove(obs)
                self.kp_to_point.pop((im, kp), None)

            if len(pt.correspondences) < 2:
                bad_point_ids.append(pid)

        # delete useless points
        for pid in bad_point_ids:
            del self.pts[pid]

        print(f"[INFO] Pruned {len(bad_point_ids)} observations above reprojection threshold ({reproj_thresh}px)")
        return len(bad_point_ids)

    def pruneGlobalObservations(self, reproj_thresh=2.0):
        bad_point_ids = []

        for pid, pt in list(self.pts.items()):
            X = pt.coord
            to_remove = []

            for obs in pt.correspondences:
                img_idx, kp_idx = obs
                img = self.images[img_idx]

                u = img.kp[kp_idx].pt
                err = reprojectionError(self.K, img.R, img.t, X, u)

                if err > reproj_thresh:
                    to_remove.append(obs)

            for obs in to_remove:
                img_idx, kp_idx = obs
                pt.correspondences.remove(obs)
                self.kp_to_point.pop((img_idx, kp_idx), None)

            # point no longer valid
            if len(pt.correspondences) < 2:
                bad_point_ids.append(pid)

        for pid in bad_point_ids:
            del self.pts[pid]

        print(f"[INFO] Pruned {len(bad_point_ids)} observations above reprojection threshold ({reproj_thresh}px)")
        return len(bad_point_ids)

    def pruneSpatialOutliers(self, k=5.0):
        """
        Removes outliers that are greater than k standard deviatuibs from centroid
        """
        if len(self.pts) < 5:
            return 0 # too few points to analyze

        coords = np.array([pt.coord for pt in self.pts.values()])
        centroid = np.mean(coords, axis=0)
        dists = np.linalg.norm(coords - centroid, axis=1)
        mean_d = np.mean(dists)
        std_d  = np.std(dists)
        if std_d < 1e-6:
            return 0

        cutoff = mean_d + k * std_d

        # find outlier point IDs
        to_delete = []
        for (pid, pt), dist in zip(self.pts.items(), dists):
            if dist > cutoff:
                to_delete.append(pid)

        # remove points + their kp_to_point references
        for pid in to_delete:
            pt = self.pts[pid]
            for (img_idx, kp_idx) in pt.correspondences:
                self.kp_to_point.pop((img_idx, kp_idx), None)
            del self.pts[pid]

        print(f"[INFO] Pruned {len(to_delete)} spatial outlier points (k={k})")
        return len(to_delete)


# @profile
def buildCorrespondences(sfm, new_img_idx):
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
        if pm is None:
            continue

        kp_pairs_tent = pm["kp_pairs_tentative"]
        for (kpA, kpB) in kp_pairs_tent:
            if k < new_img_idx:
                kp_k = kpA
                kp_new = kpB
            else:
                kp_k = kpB
                kp_new = kpA

            if kp_new in used_new_kps:
                continue
            used_new_kps.add(kp_new)

            pt_id = sfm.kp_to_point.get((k, kp_k))
            if pt_id is None:
                continue

            pt = sfm.getPoint(pt_id)
            if pt is None or pt.coord is None:
                continue

            obj_points.append(pt.coord)
            img_points.append(new_img.kp[kp_new].pt)

    if len(obj_points) == 0:
        return (np.empty((0, 3), np.float32), np.empty((0, 2), np.float32))

    return (np.asarray(obj_points, np.float32), np.asarray(img_points, np.float32))

def chooseNextImage(sfm, registered, failed, min_correspondences=6, w_existing=0.65, w_new=0.25, w_geo=0.10):
    best_j = None
    best_score = -1
    for j, img_j in sfm.images.items():
        if j in registered or j in failed or img_j.des is None:
            continue

        M_existing = 0
        M_new = 0
        M_geo = 0

        for i in registered:
            key = (min(i, j), max(i, j))
            pm = sfm.pairMatches.get(key)
            if pm is None:
                continue
            kp_tent = pm["kp_pairs_tentative"]
            kp_geo  = pm["kp_pairs_inliers"]
            M_geo += len(kp_geo)
            for (a, b) in kp_tent:
                kp_i, kp_j = (a, b) if i < j else (b, a)
                if (i, kp_i) in sfm.kp_to_point:
                    M_existing += 1 # existing observation
                if (j, kp_j) not in sfm.kp_to_point:
                    M_new += 1 # new structure

        if M_existing < min_correspondences:
            continue

        score = (w_existing * M_existing) + (w_new * M_new) + (w_geo * M_geo)

        # Track best image
        if score > best_score:
            best_score = score
            best_j = j
    return best_j

def pickInitialPair(sfm):
    """
    Choose a seed pair based on geometric inliers.
    """
    best_pair = None
    best_inliers = 0

    for key, pm in sfm.pairMatches.items():
        inliers = pm.get("kp_pairs_inliers", [])
        n_inliers = len(inliers)
        if n_inliers > best_inliers:
            best_inliers = n_inliers
            best_pair = key

    if best_pair is None:
        return (None, None), 0

    i0, j0 = best_pair
    print(f"[INFO] Selected seed pair {best_pair} using {best_inliers} geometric inliers.")
    return (i0, j0), best_inliers

@profile
def triangulateWithExistingCameras(sfm, new_idx, min_parallax_deg=1.5, reproj_thresh=1.0):
    img_j = sfm.images[new_idx]

    for i, img_i in sfm.images.items():
        if i == new_idx or not img_i.triangulated:
            continue
        if img_i.R is None or img_i.t is None or img_j.R is None or img_j.t is None:
            continue
        key = (min(i, new_idx), max(i, new_idx))
        pm = sfm.pairMatches.get(key)
        if pm is None:
            continue

        for (a, b) in pm["kp_pairs_tentative"]:
            # Map a,b to the correct image indices
            if i < new_idx:
                kp_i = a
                kp_j = b
            else:
                kp_i = b
                kp_j = a

            # if both observations are already in tracks, skip
            already_i = (i, kp_i) in sfm.kp_to_point
            already_j = (new_idx, kp_j) in sfm.kp_to_point
            if already_i and already_j:
                continue

            u_i = img_i.kp[kp_i].pt
            u_j = img_j.kp[kp_j].pt

            X = triangulatePoint(u_i, u_j,
                                 img_i.R, img_i.t,
                                 img_j.R, img_j.t,
                                 sfm.K)

            # Cheirality
            depth_i = pointDepth(img_i.R, img_i.t, X)
            depth_j = pointDepth(img_j.R, img_j.t, X)
            if depth_i <= 0 or depth_j <= 0:
                continue

            # Parallax
            par = parallaxAngle(img_i.R, img_i.t, img_j.R, img_j.t, X)
            if par < min_parallax_deg:
                continue

            # Reprojection
            err_i = reprojectionError(sfm.K, img_i.R, img_i.t, X, u_i)
            err_j = reprojectionError(sfm.K, img_j.R, img_j.t, X, u_j)
            if err_i > reproj_thresh or err_j > reproj_thresh:
                continue

            # Merge
            existing_pt_id = sfm.kp_to_point.get((i, kp_i)) or sfm.kp_to_point.get((new_idx, kp_j))
            if existing_pt_id is not None:
                pt = sfm.getPoint(existing_pt_id)
                if pt is not None and not already_j:
                    pt.addCorrespondence(new_idx, kp_j, sfm)
                continue

            sfm.addPoint(X, i, kp_i, new_idx, kp_j)

def estimateInitialRelPose(sfm, i, j):
    key = (min(i, j), max(i, j))
    pm = sfm.pairMatches.get(key)

    inlier_pairs = pm["kp_pairs_inliers"]
    if len(inlier_pairs) < 8:
        return None, None, None

    imgA = sfm.images[i]
    imgB = sfm.images[j]

    ptsA = np.float32([imgA.kp[a].pt for (a, b) in inlier_pairs])
    ptsB = np.float32([imgB.kp[b].pt for (a, b) in inlier_pairs])

    E, _ = cv.findEssentialMat(
        ptsA, ptsB, sfm.K,
        method=cv.RANSAC,
        prob=0.999,
        threshold=1.0
    )

    if E is None:
        return None, None, None

    _, R, t, _ = cv.recoverPose(E, ptsA, ptsB, sfm.K)

    imgB.setPose(R, t)
    return R, t, inlier_pairs

def triangulateInitialPoints(sfm, i, j, R, t):
    img_i = sfm.getImage(i)
    img_j = sfm.getImage(j)
    img_i.setPose(np.eye(3, dtype=np.float32), np.zeros((3,1), dtype=np.float32))

    key = (min(i, j), max(i, j))
    pm = sfm.pairMatches[key]

    for (a, b) in pm["kp_pairs_inliers"]:
        ptA = np.float32(img_i.kp[a].pt)
        ptB = np.float32(img_j.kp[b].pt)

        X = triangulatePoint(ptA, ptB, img_i.R, img_i.t, R, t, sfm.K)
        sfm.addPoint(X, i, a, j, b)

    img_i.triangulated = True
    img_j.triangulated = True



@profile
def sfmRun(dataset, viewer, matcher="BF"):
    """
    Assumptions:
        - Calibrated SfM (no fundamental matrix needed, directly calculate essential matrix with intrinsics)
        - Incremental (not global SfM)
        - Single camera per dataset
    """
    colmap = runColmap(dataset.image_dir, dataset.colmap_dir)
    if dataset.K is None:
        cam = next(iter(colmap.cameras.values()))
        dataset.K = getIntrinsics(cam.model.name, cam.params)

    sfm = SfmData()
    sfm.setExtractor(FeatureExtractor("SIFT", nfeatures=10000, contrastThreshold=0.03))
    sfm.setMatcher(FeatureMatcher(matcher, ratio=0.75))
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
            pm = sfm.genSIFTMatchPairs(img_i, img_j)
            print(f"[INFO] Matched pair ({i}, {j}): {len(pm["kp_pairs_tentative"])} tentative matches, {len(pm["kp_pairs_inliers"])} geometrically consistent")

    # ==========================================================
    # 3. Iterative triangulation and addition to point cloud
    # ==========================================================    
    (i0, j0), n_matches = pickInitialPair(sfm)
    if (i0, j0) is None:
        raise RuntimeError("Cannot find a valid initialization pair.")

    print(f"[INFO] Global seed pair ({i0}, {j0}) with {n_matches} matches")

    # esetimate first relative pose + then triangulate
    R_ij, t_ij, inlier_pairs = estimateInitialRelPose(sfm, i0, j0)
    triangulateInitialPoints(sfm, i0, j0, R_ij, t_ij)

    viewer.updateCameraPoses(sfm.getCameraData())
    viewer.updatePoints(*sfm.getPointCloud())

    registered = {i0, j0}
    failed_images = set()
    triangulatedCount = 2
    print(f"[INFO] Starting incremental reconstruction with {len(sfm.images)} images.")
    while True:
        next_idx = chooseNextImage(sfm, registered, failed_images)
        if next_idx is None:
            print("[INFO] No more images to register.")
            break

        img_j = sfm.getImage(next_idx)

        obj_points, img_points = buildCorrespondences(sfm, next_idx)
        if obj_points.shape[0] < 12:
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
            reprojectionError=0.003 * max(dataset.im_height, dataset.im_width),
            confidence=0.999
        )

        if (not success) or inliers_pnp is None or len(inliers_pnp) < 8:
            print(f"[WARN] PnP failed for image {next_idx}")
            failed_images.add(next_idx)
            continue

        inlier_ratio = len(inliers_pnp) / len(obj_points)
        if inlier_ratio < 0.25:
            print(f"[WARN] Low PnP inlier ratio ({inlier_ratio:.2f}) for image {next_idx}")
            failed_images.add(next_idx)
            continue

        R_j, _ = cv.Rodrigues(rvec)
        img_j.setPose(R_j.astype(np.float32), tvec.reshape(3, 1).astype(np.float32))
        img_j.triangulated = True
        registered.add(next_idx)
        triangulatedCount += 1

        print(f"[INFO] Image {next_idx} registered with {len(inliers_pnp)} PnP inliers.")

        sfm.pruneObservationsForNewCamera(next_idx)
        triangulateWithExistingCameras(sfm, next_idx)

        if triangulatedCount >= 4 and triangulatedCount % 3 == 0:
            print("[INFO] Running bundle adjustment...")
            runBundleAdjustment(sfm, min_points=30, verbose=0)
            num_before = len(sfm.pts)
            pruned = sfm.pruneGlobalObservations()
            pruned += sfm.pruneSpatialOutliers()
            if pruned / max(num_before, 1) >= 0.05: # if we pruned over 5% of the point cloud, BA again 
                runBundleAdjustment(sfm, min_points=30, verbose=0)

        viewer.updatePoints(*sfm.getPointCloud())
        viewer.updateCameraPoses(sfm.getCameraData())

    if triangulatedCount >= 5:
        print("[INFO] Running final bundle adjustment...")
        runBundleAdjustment(sfm, min_points=30, verbose=0)
        num_before = len(sfm.pts)
        pruned = sfm.pruneGlobalObservations()
        pruned += sfm.pruneSpatialOutliers()
        if pruned / max(num_before, 1) >= 0.05: # if we pruned over 5% of the point cloud, BA again 
            runBundleAdjustment(sfm, min_points=30, verbose=0)

    print(f"[INFO] Point cloud construction completed with {triangulatedCount} cameras registered.")
    print("\n[INFO] Running final evaluation vs GT...")
    metrics = evaluateAccuracy(sfm, colmap, print_summary=True)

def askUserForFeatureMatcher():
    options = ["BF", "FLANN"]
    print("\nSelect feature matcher:")
    for idx, name in enumerate(options):
        print(f"  {idx}: {name}")
    while True:
        user_in = input("Enter matcher number (default = BF): ").strip()
        if user_in == "":
            print("-> Using default matcher: BF")
            return "BF"
        if user_in.isdigit():
            idx = int(user_in)
            if 0 <= idx < len(options):
                print(f"-> Selected matcher: {options[idx]}")
                return options[idx]
        print("[ERROR] Invalid selection. Please try again.\n")

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
    matcher = askUserForFeatureMatcher()
    viewer = PointCloudViewer(f"SfM (scene: {os.path.basename(os.path.normpath(dataset.scene_dir))}, matcher: {matcher})")
    threading.Thread(target=sfmRun, args=(dataset, viewer, matcher), daemon=True).start()
    viewer.run()
