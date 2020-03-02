import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple


def histogram(img):
    h, w        = img.shape
    bottom_half = img[h//2:,:]
    histogram   = np.sum(bottom_half, axis=0)
    midpoint    = np.int(histogram.shape[0]//2)
    left        = np.argmax(histogram[:midpoint])
    right       = np.argmax(histogram[midpoint:]) + midpoint
    return histogram, left, right


def non_zero_idx(img):
    nonzero  = img.nonzero()
    y        = np.array(nonzero[0])
    x        = np.array(nonzero[1])
    return nonzero, x, y
    

class Polynomial:

    @classmethod
    def from_coefficients(cls, coefficients, y_meters = 15/720, x_meters = 3.7/900):
        polynomial = cls(y_meters, x_meters)
        polynomial.coef = coefficients
        return polynomial

    @classmethod
    def from_history(cls, polynomials):
        y_meters = polynomials[0].y_meters
        x_meters = polynomials[0].x_meters
        coef = np.mean([p.coef for p in polynomials], axis = 0)
        return cls.from_coefficients(coef, y_meters, x_meters)
        
    def __init__(self, y_meters = 15/720, x_meters = 3.7/900):
        self.y_meters = y_meters
        self.x_meters = x_meters

    def fit(self, x, y):
        self.coef = np.polyfit(y, x, 2)        
        
    def real_world(self, height, width):
        y, x = self.curve(height)
        self.coef_meters = np.polyfit(y * self.y_meters, x * self.x_meters, 2)  
        c1 = self.coef_meters[0]
        c2 = self.coef_meters[1]        
        curvature = 2 * c1 * height * self.y_meters + c2
        curvature = (1 + (curvature**2)) ** 1.5
        curvature = curvature / np.absolute(2 * c1)
        max_x = np.max(x)
        return curvature, max_x
    
    def curve(self, height):
        c1 = self.coef[0]
        c2 = self.coef[1]
        c3 = self.coef[2]
        y = np.linspace(0, height - 1, height)
        x = c1 * y**2 + c2 * y + c3
        return y, x
    
    
class Window(namedtuple("Window", "xlo xhi ylo yhi")):

    def filter(self, idx, idy):
        
        return (idx >= self.xlo) & (idx < self.xhi) & (idy >= self.ylo) & (idy <  self.yhi)
    
    def draw(self, img):
        cv2.rectangle(
            img,
            (self.xlo, self.ylo),
            (self.xhi, self.yhi),
            [0,255,0], 2)


class SlidingWindowDetector(
        namedtuple("SlidingWindowDetector", "nwindows margin minpix")):

    def win_height(self, height):
        return height // self.nwindows

    def window_x(self, x):
        lo = x - self.margin
        hi = x + self.margin
        return lo, hi

    def window_y(self, y, height, win_h):
        lo = height - (y + 1) * win_h
        hi = height - y       * win_h
        return lo, hi
    
    def detect(self, img):
        height                        = img.shape[0] 
        hist, left_start, right_start = histogram(img)
        nonzero, nonzero_x, nonzero_y = non_zero_idx(img)
        win_h                         = self.win_height(height)
        debug                         = np.dstack([img, img, img]) * 255        
        lx_current = left_start
        rx_current = right_start
        left_idx   = []
        right_idx  = []
        for win in range(self.nwindows):
            y_lo, y_hi     = self.window_y(win, height, win_h)
            rx_lo, rx_hi   = self.window_x(rx_current)
            lx_lo, lx_hi   = self.window_x(lx_current)
            left_window    = Window(lx_lo, lx_hi, y_lo, y_hi) 
            right_window   = Window(rx_lo, rx_hi, y_lo, y_hi) 

            left_window.draw(debug)
            right_window.draw(debug)
            
            detected_left  = left_window.filter(nonzero_x, nonzero_y).nonzero()[0]
            detected_right = right_window.filter(nonzero_x, nonzero_y).nonzero()[0]
            left_idx.append(detected_left)
            right_idx.append(detected_right)            
            if len(detected_left) > self.minpix:
                lx_current = np.int(np.mean(nonzero_x[detected_left]))
            if len(detected_right) > self.minpix:
                rx_current = np.int(np.mean(nonzero_x[detected_right]))

        left_idx   = np.concatenate(left_idx)
        right_idx  = np.concatenate(right_idx)
        leftx      = nonzero_x[left_idx]
        lefty      = nonzero_y[left_idx] 
        rightx     = nonzero_x[right_idx]
        righty     = nonzero_y[right_idx]
        left_poly  = Polynomial()
        right_poly = Polynomial()
        left_poly.fit(leftx, lefty)
        right_poly.fit(rightx, righty)
        return left_poly, right_poly, debug

    
