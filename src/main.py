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

from point_cloud_viewer import PointCloudViewer
from feature_extractor import FeatureExtractor
from feature_matcher import FeatureMatcher
from image_data import ImageData
from point_data import PointData
from utils import estimatePose, triangulate, createIntrinsicsMatrix

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
        
    # Add image given the path
    # will set image index, and compute keypoint and descriptors
    def addImage(self, image_path):
        img = ImageData()
        
        img.idx = self.imageCount
        self.imageCount += 1
        
        img.img = cv.imread(image_path)
        img.img_gray = cv.cvtColor(img.img, cv.COLOR_BGR2GRAY)
        
        kp, des = self.extractor.detectAndCompute(img.img_gray)
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
        
        #correspondences should look like [ [image1_idx, keypoint1_idx], [image2_idx, keypoint2_idx]...
        # Store point index with each associated image
        for img_idx, kp_idx in correspondences:
            pt.addCorrespondence(img_idx, kp_idx)
        
        self.pts[pt.idx] = pt
        return pt.idx
    
    def getPoint(self, pt_index):
        return self.pts.get(pt_index, None)

    def setExtractor(self, extractor):
        self.extractor = extractor
                
    def setMatcher(self, matcher):
        self.matcher = matcher

def sfmRun(datasetDir, viewer):
    """
    Assumptions:
        - Calibrated SfM (no fundamental matrix needed, directly calculate essential matrix with intrinsics)
        - Incremental (not global SfM)
    """
    sfm = SfmData()
    sfm.setExtractor(FeatureExtractor("SIFT", nfeatures=5000, contrastThreshold=0.03))
    sfm.setMatcher(matcher = FeatureMatcher("FLANN"))
    
    # Construct camera matrix
    # This block is how the github dataset constructs the camera matrix.
    # We can prob manipulate these variables for any other camera/dataset we have
    im_width = 1936
    im_height = 1296
    focal_length_35mm = 43.0  # from the EXIF data
    focal_length = max(im_width, im_height) * focal_length_35mm / 35.0
    sfm.K = np.array([[focal_length, 0, im_width / 2], [0, focal_length, im_height / 2], [0, 0, 1]])

    img1_path = os.path.normpath(os.path.join(datasetDir, "DSC_0480.JPG"))
    img2_path = os.path.normpath(os.path.join(datasetDir, "DSC_0481.JPG"))

    img1_idx = sfm.addImage(img1_path)
    img2_idx = sfm.addImage(img2_path)

    # Extract key points (for every image)
    img1 = sfm.getImage(img1_idx)
    img2 = sfm.getImage(img2_idx)
    
    kp1 = img1.kp
    des1 = img1.des
    kp2 = img2.kp
    des2 = img2.des
    print(f"Image 1: {len(kp1)} keypoints, Descriptors shape: {des1.shape}")
    print(f"Image 2: {len(kp2)} keypoints, Descriptors shape: {des2.shape}")

    # Match key points (for every pair of images)
    matches = sfm.matcher.match(des1, des2)

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    # Calculate essential matrix (needs to be done for every pair of images)
    E, mask_E = cv.findEssentialMat(pts1, pts2, sfm.K, method=cv.RANSAC, prob=0.999, threshold=1.0)

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

    # Visualize (we can call this on each iteration
    viewer.addCameraPose(img1.R, img1.t, sfm.K)
    viewer.addCameraPose(img2.R, img2.t, sfm.K)
    viewer.addPoints(ptCloud)

if __name__ == "__main__":
    viewer = PointCloudViewer("SfM Point Cloud")
    datasetDir = os.path.join(os.getcwd(),"../dataset/")
    threading.Thread(target=sfmRun, args=(datasetDir, viewer), daemon=True).start()
    viewer.run()
