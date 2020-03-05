import cv2
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from collections import namedtuple


class BirdsEyeView:
    SRC = np.float32([
        [ 579, 460],
        [ 704, 460],
        [ 222, 690],
        [1062, 690]
    ])
    DST = np.float32([
        [ 222,  100],
        [1062,  100],
        [ 222,  690],
        [1062,  690]
    ])
    
    def __init__(self, width, height, offset = 100):
        self.fwd = cv2.getPerspectiveTransform(BirdsEyeView.SRC, BirdsEyeView.DST)
        self.bwd = cv2.getPerspectiveTransform(BirdsEyeView.DST, BirdsEyeView.SRC)
        
    def project(self, img):
        h = img.shape[0]
        w = img.shape[1]
        return cv2.warpPerspective(
            img, self.fwd, (w, h))

    def backward(self, img):
        h = img.shape[0]
        w = img.shape[1]
        return cv2.warpPerspective(
            img, self.bwd, (w, h))
