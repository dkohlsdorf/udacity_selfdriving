import numpy as np
import cv2 
import os 
import matplotlib.pyplot as plt
from collections import namedtuple


CALIBRATION_PREFIX = "calibration"
IMG_ENDING = ".jpg"


def is_calibration_data(filename):    
    is_img   = filename.endswith(IMG_ENDING)
    is_calib = filename.startswith(CALIBRATION_PREFIX)
    return is_img and is_calib


def corners(img_path, nx, ny):
    img  = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
    if ret:
        return corners, gray.shape[::-1]
    
        
def calibration_points(calibration_folder, nx, ny):    
    object_points        = np.zeros((nx * ny, 3), np.float32)
    object_points[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)            
    for filename in os.listdir(calibration_folder):
        if is_calibration_data(filename):
            path = "{}/{}".format(calibration_folder, filename)
            detected_corners = corners(path, nx, ny)
            if detected_corners is not None:
                corner_points, shape = detected_corners
                yield shape, corner_points, object_points
                
                
class CalibratedCamera(namedtuple("CalibratedCamera", "camera_mat distortion rotation translation")):

    @classmethod
    def from_images(cls, calibration_folder, nx = 9, ny = 6):
        object_points = []
        image_points  = []
        shape         = None
        for s, c, o in calibration_points(calibration_folder, nx, ny):
            shape = s
            object_points.append(o)
            image_points.append(c)
        ret, camera_mat, distortion, rotation, translation = cv2.calibrateCamera(
            object_points, image_points, shape, None, None)
        return cls(camera_mat, distortion, rotation, translation)

    def undistort(self, img):
        return cv2.undistort(img, self.camera_mat, self.distortion, None, self.camera_mat)
    
