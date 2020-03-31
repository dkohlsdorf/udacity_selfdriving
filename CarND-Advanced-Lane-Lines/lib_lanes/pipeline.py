import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import sys

from moviepy.editor import VideoFileClip

from lib_lanes.cameras import CalibratedCamera
from lib_lanes.thresholding import BinaryConverter
from lib_lanes.perspective import BirdsEyeView
from lib_lanes.lane_detector import SlidingWindowDetector
from lib_lanes.smooth import SmoothedPolynomial

class Pipeline:

    def __init__(self, logfile = "log.csv"):
        self.cam          = CalibratedCamera.from_images("camera_cal")
        self.binary_cvt   = BinaryConverter(120, 255, 50, 250)
        self.projection   = BirdsEyeView(720, 1280, 100)
        self.windowing    = SlidingWindowDetector(9, 100, 50) 
        self.left_smooth  = SmoothedPolynomial(25)
        self.right_smooth = SmoothedPolynomial(25) 
        self.log          = open(logfile, "w")
        self.log.write("curvature, distance\n")

    def done(self):
        self.log.close()
        
    def process_image(self, img, return_debug = False):
        h, w, _ = img.shape
        undistorted = self.cam.undistort(img)
        binary      = self.binary_cvt.convert(undistorted)
        bird        = self.projection.project(binary)

        left, right, debug = self.windowing.detect(bird)
        left               = self.left_smooth.polynomial(w, h, left)
        right              = self.right_smooth.polynomial(w, h, right)

        ly, lx = left.curve(720)
        leftp  = np.int32(np.stack([lx, ly]).T)
        ry, rx = right.curve(720)
        rightp = np.int32(np.stack([rx, ry]).T)

        overlay = np.zeros_like(debug).astype(np.uint8)
        points  = np.vstack((leftp, np.flipud(rightp)))
    
        cv2.polylines(debug, [leftp],  False, (255, 0, 0), 20)
        cv2.polylines(debug, [rightp], False, (255, 0, 0), 20)
    
        lc, lmax_x = left.real_world(h, w)
        rc, rmax_x = right.real_world(h, w)
    
        curvature    = int(lc + rc) // 2
        
        center_lane        = np.copy(leftp)
        center_lane[:, 0] += np.int32(rmax_x - lmax_x) // 2

        center_dst        = np.copy(leftp)
        center_dst[:, 0] += np.int32(BirdsEyeView.DST[-1][0] -
                                     BirdsEyeView.DST[0][0]) // 2
    
        offset = np.sqrt(np.mean(np.square(center_lane - center_dst)))
    
        alpha = 0.6
        overlay = np.zeros_like(undistorted).astype(np.uint8)
        cv2.fillPoly(overlay, [points], (0, 255, 0))
        cv2.polylines(overlay, [center_lane], False, (255, 0, 0), 20)
        cv2.polylines(overlay, [center_dst],  False, (255, 255, 255), 20)
        cv2.polylines(overlay, [leftp],   False, (255, 0, 0), 20)
        cv2.polylines(overlay, [rightp],  False, (255, 0, 0), 20)    
        overlay = self.projection.backward(overlay)    
        marked = cv2.addWeighted(undistorted, alpha, overlay, 1 - alpha, 0)        
        curve_text = "curvature {} distance {:0.3}".format(
            curvature,
            offset * left.x_meters
        )    
        cv2.putText(marked, curve_text, (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 4)

        self.log.write("{},{}\n".format(curvature, offset * left.x_meters))
        if return_debug:
            return debug
        return marked

