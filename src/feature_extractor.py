import cv2 as cv

class FeatureExtractor:
    def __init__(self, method="SIFT", **kwargs):
        self.method = method.upper()
        self.extractor = self._create_extractor(**kwargs)

    def _create_extractor(self, **kwargs):
        if self.method == "SIFT":
            return cv.SIFT_create(**kwargs)
        elif self.method == "ORB":
            return cv.ORB_create(**kwargs)
        elif self.method == "AKAZE":
            return cv.AKAZE_create(**kwargs)
        elif self.method == "BRISK":
            return cv.BRISK_create(**kwargs)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def detectAndCompute(self, image):
        """Detect keypoints and compute descriptors."""
        return self.extractor.detectAndCompute(image, None)

    def drawKeypoints(self, image, keypoints, color=(0, 255, 0)):
        """Draw detected keypoints on the image."""
        return cv.drawKeypoints(
            image,
            keypoints,
            None,
            color=color,
            flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )
