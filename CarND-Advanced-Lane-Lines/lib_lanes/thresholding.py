import numpy as np
import cv2
from collections import namedtuple


class BinaryConverter(namedtuple("BinaryImage", "s_lo s_hi x_lo x_hi")):

    def binary_sobel_x(self, img):
        gray         = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        sobelx       = cv2.Sobel(gray, cv2.CV_64F, 1, 0) 
        abs_sobelx   = np.absolute(sobelx) 
        scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
        sxbinary     = np.zeros_like(scaled_sobel)
        sxbinary[(scaled_sobel >= self.x_lo) & (scaled_sobel <= self.x_hi)] = 1
        return sxbinary

    def binary_hls(self, img):
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        s_channel = hls[:,:,2]                        
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= self.s_lo) & (s_channel <= self.s_hi)] = 1
        return s_binary
    
    def convert(self, img):
        s_binary = self.binary_hls(img)
        x_binary = self.binary_sobel_x(img)
        combined_binary = np.zeros_like(x_binary)
        combined_binary[(s_binary == 1) | (x_binary == 1)] = 1
        return combined_binary
    
