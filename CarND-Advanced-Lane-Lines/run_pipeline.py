import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from lib_lanes.cameras import CalibratedCamera
from lib_lanes.thresholding import BinaryConverter
from lib_lanes.perspective import BirdsEyeView
from lib_lanes.lane_detector import SlidingWindowDetector


img         = mpimg.imread("test_images/straight_lines1.jpg")

cam        = CalibratedCamera.from_images("camera_cal")
binary_cvt = BinaryConverter(170, 255, 20, 100)
projection = BirdsEyeView(img.shape[1], img.shape[0], 100)
windowing  = SlidingWindowDetector(9, 100, 50) 

undistorted = cam.undistort(img)
binary      = binary_cvt.convert(undistorted)
bird        = projection.project(binary)

_, _, _, _, debug = windowing.detect(bird)

plt.imshow(debug)
plt.show()
