import numpy as np

class PointData:
    def __init__(self):
        self.idx = None
        # [X,Y,Z]
        self.coord = None
        # List of pairs [ [image1_idx, keypoint1_idx], [image2_idx, keypoint2_idx]...]
        self.correspondences = []
        self.color = None # vec3 - RGB

    def addCorrespondence(self, image_idx, keypoint_idx, sfm=None):
        if [image_idx, keypoint_idx] not in self.correspondences:
            self.correspondences.append([image_idx,keypoint_idx])
            if sfm is not None:
                self.updateColor(sfm)

    def updateColor(self, sfm):
        if len(self.correspondences) == 0:
            return

        colors = []
        for img_idx, kp_idx in self.correspondences:
            img = sfm.images.get(img_idx)
            if img is None or img.img is None:
                continue

            x, y = map(int, img.kp[kp_idx].pt)
            if (0 <= x < img.img.shape[1] and
                0 <= y < img.img.shape[0]):
                b,g,r = img.img[y, x]  # BGR from cv2
                colors.append([r/255.0, g/255.0, b/255.0])

        if len(colors) > 0:
            self.color = np.mean(colors, axis=0).astype(np.float32)

