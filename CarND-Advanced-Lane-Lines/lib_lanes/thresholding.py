import matplotlib.pyplot as plt
import numpy as np
import cv2
from collections import namedtuple


class BinaryConverter(namedtuple("BinaryImage", "s_lo s_hi x_lo x_hi")):
    
    def binary_sobel(self, img, kernel=3):
        gray         = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        sobelx       = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel)
        absx         = np.abs(sobelx)
        abs_scaled   = np.uint8(255 * absx / np.max(absx))        
        binary = np.zeros_like(abs_scaled)
        binary[(abs_scaled >= self.x_lo) & (abs_scaled <= self.x_hi)] = 1
        return binary

    def binary_hls(self, img):
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        s_channel = hls[:,:,2]                        
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= self.s_lo) & (s_channel <= self.s_hi)] = 1
        return s_binary
    
    def convert(self, img):
        s_binary = self.binary_hls(img)
        binary = self.binary_sobel(img)        
        combined_binary = np.zeros_like(binary)
        combined_binary[(s_binary == 1) | (binary == 1)] = 1
        return combined_binary
    
