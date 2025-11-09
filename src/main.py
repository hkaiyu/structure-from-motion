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

from point_cloud_viewer import PointCloudViewer
from feature_extractor import FeatureExtractor
from feature_matcher import FeatureMatcher
from image_data import ImageData
from point_data import PointData
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
        # TODO: Maintain 2D-3D association maps for track management (used by incremental SfM):
        #   - self.kp_to_point = {}  # (img_idx, kp_idx) -> point_id
        #   - self.point_observations = {}  # point_id -> list of (img_idx, kp_idx)
        #   - Update these when triangulating and when adding observations for existing points

    def setCameraIntrinsics(self, focal_length_35mm, im_width, im_height):
        focal_length = max(im_width, im_height) * focal_length_35mm / 35.0
        self.K = np.array([[focal_length, 0, im_width / 2], [0, focal_length, im_height / 2], [0, 0, 1]])

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
    
    # add point, will add 3d point and relevant correspondences
    def addPoint(self, coord, correspondences):
        pt = PointData()
        
        pt.idx = self.pointCount
        self.pointCount += 1
        
        pt.coord = coord
        
        # correspondences should look like [ [image1_idx, keypoint1_idx], [image2_idx, keypoint2_idx] ... ]
        # Store point index with each associated image
        for img_idx, kp_idx in correspondences:
            pt.addCorrespondence(img_idx, kp_idx)
            # TODO: When association maps are added to SfmData, also update:
            #   self.kp_to_point[(img_idx, kp_idx)] = pt.idx
            #   self.point_observations.setdefault(pt.idx, []).append((img_idx, kp_idx))
        
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


def sfmRun(datasetDir, viewer):
    """
    Assumptions:
        - Calibrated SfM (no fundamental matrix needed, directly calculate essential matrix with intrinsics)
        - Incremental (not global SfM)
    """
    sfm = SfmData()
    # sfm.setExtractor(FeatureExtractor("SIFT", nfeatures=5000, contrastThreshold=0.03))
    # sfm.setMatcher(matcher = FeatureMatcher("FLANN"))
    sfm.setExtractor(FeatureExtractor("SIFT", nfeatures=500))
    sfm.setMatcher(matcher = FeatureMatcher("BF", norm=cv.NORM_L2, crossCheck=False))

    # Construct camera matrix
    # This block is how the github dataset constructs the camera matrix.
    # We can prob manipulate these variables for any other camera/dataset we have
    im_width = 1936
    im_height = 1296
    focal_length_35mm = 43.0  # from the EXIF data
    sfm.setCameraIntrinsics(focal_length_35mm, im_width, im_height)

    # ==========================================================
    # 1. Load images and extract features
    # ==========================================================
    images = load_all_images(datasetDir)
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
    

    
    # Iterative Section from here
    
    # Calculate essential matrix (needs to be done for every pair of images)
    E, mask_E = cv.findEssentialMat(pts1, pts2, sfm.K, method=cv.RANSAC, prob=0.999, threshold=1.0)
    # TODO: Check for degenerate/insufficient inliers and skip/choose another initial pair when needed.
    # TODO: Consider normalizing points (undistorting and using normalized coordinates) before E estimation.

    mask_E = mask_E.ravel().astype(bool) # if you don't do this, opencv complains
    inliers1 = pts1[mask_E]
    inliers2 = pts2[mask_E]

    # The first iteration of SfM treats first camera as base projection so P = [I | 0] (R = np.eye(3), t = [0, 0, 0])
    img1.R = np.eye(3)
    img1.t = np.zeros((3, 1))
    R, t, mask_P = estimatePose(inliers1, inliers2, E, sfm.K)
    img2.setPose(R,t)

    mask_P = mask_P.ravel().astype(bool)
    inliers1 = inliers1[mask_P]
    inliers2 = inliers2[mask_P]
    print("Initial match count: ", mask_E.size)
    print("Final inliers count: ", mask_P.sum())

    ptCloud = triangulate(inliers1, inliers2, img1.R, img1.t, img2.R, img2.t, sfm.K)
    # TODO: Filter triangulated points by cheirality (positive depth), parallax, and reprojection error.
    # TODO: Persist valid 3D points in SfmData (addPoint) and maintain observation tracks.
    # TODO: Run initial bundle adjustment (2-view BA) to refine poses and 3D points.

    # Visualize (we can call this on each iteration
    viewer.addCameraPose(img1.R, img1.t, sfm.K)
    viewer.addCameraPose(img2.R, img2.t, sfm.K)
    viewer.addPoints(ptCloud)

    # TODO: Implement incremental reconstruction loop:
    #   - For each remaining image: find 2D-3D correspondences via feature matches + known tracks.
    #   - Estimate new camera pose with solvePnPRansac (check MIN_REQUIRED_POINTS).
    #   - Triangulate new points with existing calibrated views and add to the model.
    #   - Periodically run local/global bundle adjustment.
    #   - Update viewer after each step.
    # TODO: Save reconstruction to disk (e.g., PLY for points + JSON for camera poses).

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=None, help="Path to dataset directory (defaults to project_root/dataset)")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    dataset_path = Path(args.dataset).resolve() if args.dataset else project_root / "dataset"
    if not dataset_path.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")

    viewer = PointCloudViewer("SfM Point Cloud")
    threading.Thread(target=sfmRun, args=(str(dataset_path), viewer), daemon=True).start()
    viewer.run()
