class PointData:
    def __init__(self):
        self.idx = None
        # [X,Y,Z]
        self.coord = None
        # List of pairs [ [image1_idx, keypoint1_idx], [image2_idx, keypoint2_idx]...]
        self.correspondences = []

    def addCorrespondence(self, image_idx, keypoint_idx):
        if [image_idx, keypoint_idx] not in self.correspondences:
            self.correspondences.append([image_idx,keypoint_idx])

    # Placeholder function to get RGB value of the point
    # Should loop through correspondences to get RGB values of each corresponding keypoint, and then average them
    def getColor(self):
        return None
