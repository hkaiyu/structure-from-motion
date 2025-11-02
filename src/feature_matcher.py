import cv2 as cv

class FeatureMatcher:
  def __init__(self, method="BF", **kwargs):
        self.method = method.upper()
        self.matcher = self._create_matcher(**kwargs)

  def _create_matcher(self, **kwargs):
        if self.method == "BF":
            norm = kwargs.get("norm", cv.NORM_L2)
            crossCheck = kwargs.get("crossCheck", False)
            return cv.BFMatcher(normType=norm, crossCheck=crossCheck)
        elif self.method == "FLANN":
            # Flann needs specific params for ORB (or other binary descriptors), otherwise it can crash
            index_params = kwargs.get("index_params", dict(algorithm=0, trees=5))
            search_params = kwargs.get("search_params", dict(checks=50))
            return cv.FlannBasedMatcher(index_params, search_params)
        else:
            raise ValueError(f"Unknown matcher: {self.method}")
  def match(self, des1, des2):
        """Returns a list of matches sorted by distance."""
        matches = self.matcher.match(des1, des2)
        return sorted(matches, key=lambda x: x.distance)

  def drawMatches(self, img1, kp1, img2, kp2, matches, max_matches=50):
        """Draw the best {max_matches} matches."""
        return cv.drawMatches(img1, kp1, img2, kp2,
            matches[:max_matches], None,
            flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
