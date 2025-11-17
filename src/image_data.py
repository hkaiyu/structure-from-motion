class ImageData:
    def __init__(self):
        # Probably don't need both an index and path, just leaving both in for now to see what we want
        self.idx = None
        self.path = None
        self.img = None
        # self.img_gray = None
        # Keypoints and descriptors
        self.kp = None
        self.des = None
        # Pose info
        self.R = None
        self.t = None
        # Bool to store if we already used this image for triangulation
        self.triangulated = False

    def setPose(self, R, t):
        self.R = R
        self.t = t

    def getKeypoints(self):
        return self.kp

    def getDescriptors(self):
        return self.des
