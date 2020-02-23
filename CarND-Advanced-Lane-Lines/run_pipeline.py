import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from lib_lanes.cameras import CalibratedCamera
from lib_lanes.thresholding import BinaryConverter

cam        = CalibratedCamera.from_images("camera_cal")
binary_cvt = BinaryConverter(170, 255, 20, 100)


img = mpimg.imread("/Users/daniel.kohlsdorf/Dropbox/udacity/self_driving/bridge_shadow.jpg")
undistorted = cam.undistort(img)
binary = binary_cvt.convert(img)

plt.figure(figsize=(50, 50))
plt.subplot(2,2, 1)
plt.imshow(img)
plt.subplot(2,2, 2)
plt.imshow(undistorted)
plt.subplot(2,2, 3)
plt.imshow(undistorted - img)
plt.subplot(2,2, 4)
plt.imshow(binary, cmap='gray')
plt.savefig('test.png')
