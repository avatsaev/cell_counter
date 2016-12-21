#! /usr/local/bin/python

import numpy as np
import cv2
import argparse
import imutils

def say_hello:
    print "heloo"

def show_img(img):
    # cv2.imshow("image",img)

    screen_res = 1280, 720
    scale_width = screen_res[0] / img.shape[1]
    scale_height = screen_res[1] / img.shape[0]
    scale = min(scale_width, scale_height)
    window_width = int(img.shape[1] * scale)
    window_height = int(img.shape[0] * scale)

    cv2.namedWindow('dst_rt', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('dst_rt', window_width, window_height)

    cv2.imshow('dst_rt', img)
    cv2.waitKey(0)

def bitwise_blend(img1, img2):

    rows, cols = img2.shape
    roi = img1[0:rows, 0:cols ]

    mask_inv = cv2.bitwise_not(img2)

    img1_bg = cv2.bitwise_and(roi, roi, mask = mask_inv)
    img2_fg = cv2.bitwise_and(img2, img2, mask = img2)

    return cv2.add(img1_bg, img2_fg)

def detect_edges(img, min_val=5, max_val=22):

    min_val = float(min_val)
    max_val = float(max_val)

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Canny(gray_img, min_val,max_val)

def dynamic_blend(img1, img2):


    # I want to put logo on top-left corner, So I create a ROI
    rows,cols,ch = img2.shape
    roi = img1[0:rows, 0:cols ]

    # Now create a mask of logo and create its inverse mask also
    img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    # Now black-out the area of logo in ROI
    img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)

    # Take only region of logo from logo image.
    img2_fg = cv2.bitwise_and(img2,img2,mask = mask)

    # Put logo in ROI and modify the main image
    dst = cv2.add(img1_bg,img2_fg)
    img1[0:rows, 0:cols ] = dst

    return img1

def opening_morph(img):
    kernel = np.ones((5,5 ), np.uint8)
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

def dilate(img, iterations=1):
    kernel = np.ones((5,5),np.uint8)
    return cv2.dilate(img,kernel,iterations = iterations)


def auto_canny(image, sigma=0.33):

    # compute the median of the single channel pixel intensities

    # return the edged image
    return edged

def auto_canny_edge(image, sigma=0.30):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # apply Canny edge detection using a wide threshold, tight
    # threshold, and automatically determined threshold

    v = np.median(blurred)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    auto = cv2.Canny(blurred, lower, upper)

    return auto

def histeq(img):


    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

    # equalize the histogram of the Y channel
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])

    # convert the YUV image back to RGB format
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    #cv2.imshow('Color input image', img)
    #cv2.imshow('Histogram equalized', img_output)

    return img_output


#####################################################################################
