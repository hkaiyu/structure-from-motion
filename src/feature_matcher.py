import numpy as np
import cv2 as cv

class FeatureMatcher:
    def __init__(self, method="BF", ratio=0.75):
        self.method = method.upper()
        self.ratio = ratio

        if self.method == "BF":
            self.matcher = cv.BFMatcher(normType= cv.NORM_L2, crossCheck=False)
        elif self.method == "FLANN":
            # Flann needs specific params for ORB (or other binary descriptors), otherwise it can crash
            index_params = dict(algorithm=1, trees=5)
            search_params = dict(checks=50)
            self.matcher = cv.FlannBasedMatcher(index_params, search_params)
        else:
            raise ValueError(f"Unknown matcher: {self.method}")

    def _apply_ratio_test(self, matches):
        tentative = []
        for m, n in matches:
            if m.distance < self.ratio * n.distance:
                tentative.append(m)
        return tentative

    def _symmetric_filter(self, matchesAB, matchesBA):
        # Build reverse lookup: (trainIdx -> queryIdx)
        reverse = {(m.trainIdx, m.queryIdx) for m in matchesBA}
        mutual = [
            m for m in matchesAB
            if (m.queryIdx, m.trainIdx) in reverse
        ]
        return mutual

    def _geometric_filter(self, imgA, imgB, tentative_matches, K):
        """
        Use Essential matrix RANSAC to find geometric inliers.
        Returns list of matches that are consistent with epipolar geometry.
        """

        if len(tentative_matches) < 8:
            return []

        ptsA = np.float32([imgA.kp[m.queryIdx].pt for m in tentative_matches])
        ptsB = np.float32([imgB.kp[m.trainIdx].pt for m in tentative_matches])

        E, mask = cv.findEssentialMat(
            ptsA, ptsB, K,
            method=cv.RANSAC,
            prob=0.999,
            threshold=1.0
        )

        if mask is None:
            return []

        mask = mask.ravel().astype(bool)

        inliers = [m for m, use in zip(tentative_matches, mask) if use]
        return inliers

    def match(self, imgA, imgB, K):
        """
        returns:
            1. all matches found (no filter)
            2. tentative matches (ratio test + symmetric)
            3. geometric inliers (essential matrix)
        """

        knnAtoB = self.matcher.knnMatch(imgA.des, imgB.des, k=2)
        knnBtoA = self.matcher.knnMatch(imgB.des, imgA.des, k=2)

        full_matches = [m[0] for m in knnAtoB]  # raw matches, before any filtering

        tentAB = self._apply_ratio_test(knnAtoB)
        tentBA = self._apply_ratio_test(knnBtoA)
        tentative = self._symmetric_filter(tentAB, tentBA)

        if K is None:
            geom_inliers = []
        else:
            geom_inliers = self._geometric_filter(imgA, imgB, tentative, K)

        return full_matches, tentative, geom_inliers

  # def drawMatches(self, img1, kp1, img2, kp2, matches, max_matches=50):
  #       """Draw the best {max_matches} matches."""
  #       return cv.drawMatches(img1, kp1, img2, kp2,
  #           matches[:max_matches], None,
  #           flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
  #       )
