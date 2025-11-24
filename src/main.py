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
import dataset_loader

from point_cloud_viewer import PointCloudViewer
from feature_extractor import FeatureExtractor
from feature_matcher import FeatureMatcher
from image_data import ImageData
from point_data import PointData
from bundle_adjustment import runBundleAdjustment
from utils import estimatePose, triangulate, createIntrinsicsMatrix, load_all_images

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

    def genSIFTMatchPairs(self, img1, img2, numberOfMatches=None, use_cache=True):
        """Generate matched 2D point pairs for two images using cached or freshly
        computed descriptor matches. Returns (pts1, pts2, matches, kp1, kp2).

        - numberOfMatches: if set, returns only the top-N matches by distance
        - use_cache: if True, use self.pairMatches when available
        """
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
        matches = self.matcher.match(des1, des2)
        self.pairMatches[key] = matches

        # Sort them based on distance (dissimilarity) between two descriptors
        matches = sorted(matches, key=lambda m: m.distance)

        # Optionally truncate to top-N matches
        if numberOfMatches is not None:
            matches = matches[:numberOfMatches]

        # Build point arrays
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

        return pts1, pts2, matches, kp1, kp2

def voxelDownsampleFilter(sfm, voxel_size=0.1):
    """ Splits up world space into voxels of voxel_size. All points mapping to same voxel get grouped """
    pts = []
    for pt in sfm.pts.values():
        if pt.coord is not None:
            pts.append((pt.idx, pt.coord))

    voxels = {}
    for idx, coord in pts:
        key = tuple((coord / voxel_size).astype(int))
        if key not in voxels:
            voxels[key] = idx  # keep one point per voxel (for less cluttered view)

    keep_indices = set(voxels.values())
    for idx in list(sfm.pts.keys()):
        if idx not in keep_indices:
            del sfm.pts[idx]

    sfm.kp_to_point = {
        k: v for (k, v) in sfm.kp_to_point.items() if v in sfm.pts
    }

def sfmRun(dataset, viewer):
    """
    Assumptions:
        - Calibrated SfM (no fundamental matrix needed, directly calculate essential matrix with intrinsics)
        - Incremental (not global SfM)
    """
    sfm = SfmData()
    sfm.setExtractor(FeatureExtractor("SIFT", nfeatures=5000, contrastThreshold=0.03))
    sfm.setMatcher(matcher = FeatureMatcher("FLANN"))
    # sfm.setExtractor(FeatureExtractor("SIFT", nfeatures=5000))
    # sfm.setMatcher(matcher = FeatureMatcher("BF", norm=cv.NORM_L2, crossCheck=False))

    # Construct camera matrix
    # This block is how the github dataset constructs the camera matrix.
    # We can prob manipulate these variables for any other camera/dataset we have
    im_width =  dataset.im_width
    im_height = dataset.im_height
    sfm.setCameraIntrinsics(dataset.K)

    # ==========================================================
    # 1. Load images and extract features
    # ==========================================================
    images = load_all_images(dataset.dataset_dir)
    features = {}
    img_indices = []
    for img_id, img in enumerate(images):
        idx = sfm.addImage(img)
        img_indices.append(idx)
    print(f"Loaded {len(img_indices)} images")


    # ==========================================================
    # 2. Match features between image pairs
    # ==========================================================
    # Compute and cache pairwise matches for all unique pairs
    for i, j in combinations(img_indices, 2):
        img_i = sfm.getImage(i)
        img_j = sfm.getImage(j)
        if img_i.des is None or img_j.des is None:
            print(f"Skipping pair ({i},{j}) due to missing descriptors")
            continue
        # TODO: Improve match quality: use KNN ratio test (Lowe) and mutual consistency (cross-check)
        #       to prune ambiguous matches before geometric estimation.
        #       Consider spatial verification (e.g., using Essential/Fundamental with RANSAC) per pair.
        pts1, pts2, pair_matches, kp1, kp2 = sfm.genSIFTMatchPairs(img_i, img_j)
        sfm.pairMatches[(i, j)] = pair_matches
        
        print(f"Matched pair ({i}, {j}): {len(pair_matches)} matches")
        # print(f"Image 1: {len(kp1)} keypoints")
        # print(f"Image 2: {len(kp2)} keypoints")
    
    # ==========================================================
    # 3. Iterative triangulation and addition to point cloud
    # ==========================================================    
    
    # Init stuff for first iteration of loop
    # The first iteration of SfM treats first camera as base projection so P = [I | 0] (R = np.eye(3), t = [0, 0, 0])
    img_i = sfm.getImage(0)
    img_i.setPose(np.eye(3), np.zeros((3, 1)))
    img_i.triangulated = True
    triangulatedCount = 1
    viewer.updateCameraPoses([{
                "R": img_i.R,
                "t": img_i.t,
                "K": sfm.K,
                "name": f"Camera 0",
                "color": "orange"}])

    while triangulatedCount < sfm.imageCount:
        # Placeholder for getting the next image, for now it is just getting whatever the image at next index is
        img_j = sfm.getImage(img_i.idx+1)
        
        # Match with Lowe's ratio test
        matches = sfm.matcher.knnMatch(img_i.des, img_j.des)
        if len(matches) < 20:
            print(f"Skipping triangulation on {img_i.idx} and {img_j.idx}, bad matches")
            break
        
        pts_i = np.float32([img_i.kp[m.queryIdx].pt for m in matches])
        pts_j = np.float32([img_j.kp[m.trainIdx].pt for m in matches])
        # Need to store indices so we don't lose them after filtering
        idx_i = np.array([m.queryIdx for m in matches])
        idx_j = np.array([m.trainIdx for m in matches])
        
        # Calculate essential matrix (needs to be done for every pair of images)
        E, mask_E = cv.findEssentialMat(pts_i, pts_j, sfm.K, method=cv.RANSAC, prob=0.999, threshold=1.0)
        # TODO: Check for degenerate/insufficient inliers and skip/choose another initial pair when needed.
        # TODO: Consider normalizing points (undistorting and using normalized coordinates) before E estimation.
        if mask_E is None or mask_E.sum()<10:
            print(f"Skipping triangulation on {img_i.idx} and {img_j.idx}, bad E")
            break
        
        mask_E = mask_E.ravel().astype(bool) # if you don't do this, opencv complains
        inliers_i = pts_i[mask_E]
        inliers_j = pts_j[mask_E]
        idx_i = idx_i[mask_E]
        idx_j = idx_j[mask_E]
        
        R, t, mask_P = estimatePose(inliers_i, inliers_j, E, sfm.K)
        # Need to chain new image j pose with image i pose
        R = img_i.R @ R
        t = img_i.R @ t + img_i.t
        img_j.setPose(R,t)
        
        mask_P = mask_P.ravel().astype(bool)
        inliers_i = inliers_i[mask_P]
        inliers_j = inliers_j[mask_P]
        idx_i = idx_i[mask_P]
        idx_j = idx_j[mask_P]
        print(f"Initial match count between image {img_i.idx} and image {img_j.idx}: {mask_E.size}")
        print(f"Final inliers count between image {img_i.idx} and image {img_j.idx}: {mask_P.sum()}")
        # We will probably want to use 1 or 2 decimal points, integer looked unusable
        ptCloud = triangulate(inliers_i, inliers_j, img_i.R, img_i.t, img_j.R, img_j.t, sfm.K, decimal_pts=2)
        # idx i/j, inliers i/j, and ptCloud all should have same number of entries, can loop through any of their sizes
        for idx in range(idx_j.size):
            # TODO: Need some kind of handling on what to do if we have a 3d point that already exists due to rounding
            # Merge into one? Do suppression? Need to research

            # try merging
            pt_idx = sfm.kp_to_point.get((img_i.idx, idx_i[idx]))
            if pt_idx is None:
                pt_idx = sfm.kp_to_point.get((img_j.idx, idx_j[idx]))

            if pt_idx is not None:
                pt = sfm.getPoint(pt_idx)
                pt.addCorrespondence(img_i.idx, idx_i[idx], sfm)
                pt.addCorrespondence(img_j.idx, idx_j[idx], sfm)
                sfm.kp_to_point[(img_i.idx, idx_i[idx])] = pt.idx
                sfm.kp_to_point[(img_j.idx, idx_j[idx])] = pt.idx
            else:
                sfm.addPoint(ptCloud[idx], img_i.idx, idx_i[idx], img_j.idx, idx_j[idx])



        # TODO: Filter triangulated points by cheirality (positive depth), parallax, and reprojection error.
        runBundleAdjustment(sfm, min_points=20, verbose=1) # can pick verbose=2 here if want to print progress

        voxelDownsampleFilter(sfm)

        # Build full camera list from SfmData
        cams = []
        for img_idx, img in sfm.images.items():
            if img.R is None or img.t is None:
                continue
            cams.append({
                "R": img.R,
                "t": img.t,
                "K": sfm.K,
                "name": f"Camera {img_idx}",
                "color": "orange"
            })

        pts_xyz, colors = sfm.getPointCloud()
        viewer.updatePoints(pts_xyz, colors)
        viewer.updateCameraPoses(cams)

        # Increment, img_j becomes img_i for the next iteration
        img_j.triangulated = True
        triangulatedCount += 1

        img_i = img_j

        # TODO: Implement incremental reconstruction loop:
        #   - For each remaining image: find 2D-3D correspondences via feature matches + known tracks.
        #   - Estimate new camera pose with solvePnPRansac (check MIN_REQUIRED_POINTS).
        #   - Triangulate new points with existing calibrated views and add to the model.
        #   - Periodically run local/global bundle adjustment.
        #   - Update viewer after each step.
        # TODO: Save reconstruction to disk (e.g., PLY for points + JSON for camera poses).
        
    # For debug, I would reccommend just printing terminal output to a file if you use this
    # for pt_id, pt in sfm.pts.items():
    #     print(f"Point {pt_id}: 3D coords = {pt.coord}")
    #     for img_idx, kp_idx in pt.correspondences:
    #         print(f"  seen in image {img_idx}, keypoint {kp_idx}")
    # for (img_idx, kp_idx), pt_id in sfm.kp_to_point.items():
    #     print(f"Image {img_idx}, keypoint {kp_idx} -> Point {pt_id}, 3D = {sfm.pts[pt_id].coord}")

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    dataset_dir = Path(project_root / "dataset")
    dataset = dataset_loader.select_dataset(dataset_dir)
    print(f"Using dataset at: {dataset.dataset_dir}")
    print(f"Image Width: {dataset.im_height}")
    print(f"Image Height: {dataset.im_width}")
    print(f"Focal Length: {dataset.focal_length}\n")
    
    if not dataset:
        raise FileNotFoundError(f"Error loading the selected dataset")

    viewer = PointCloudViewer("SfM Point Cloud")
    threading.Thread(target=sfmRun, args=(dataset, viewer), daemon=True).start()
    viewer.run()
