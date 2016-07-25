#! /usr/local/bin/python

import numpy as np
import cv2
from pyimagesearch.shapedetector import ShapeDetector
import argparse
import imutils

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



print("processing edges...")


img = cv2.imread('contours_sample_1_raw.jpg')
img_copy = img.copy()

template = cv2.imread('spot_pattern_2.jpg', 0)
w, h = template.shape[::-1]


img = cv2.medianBlur(img,5)
#opening morph to remove points
kernel = np.ones((5,5 ), np.uint8)
opening_morph = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
cv2.imwrite( "output_pattern/opening_morph.png", opening_morph );



gray_img = cv2.cvtColor(opening_morph, cv2.COLOR_BGR2GRAY)
edged_img = cv2.Canny(gray_img, 5, 22)
cv2.imwrite( "output_pattern/edged.png", edged_img );


kernel = np.ones((5,5 ),np.uint8)
dilated_img  = cv2.dilate(edged_img,kernel,iterations = 2)
cv2.imwrite( "output_pattern/dilated.png", dilated_img );


res = cv2.matchTemplate(dilated_img,template,cv2.TM_CCOEFF_NORMED)


threshold = 0.19
loc = np.where( res >= threshold)


for pt in zip(*loc[::-1]):
    cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0,255,0), 3)

cv2.imwrite('output_pattern/res.png',img)


print "PATTERN MATCHING DONE"
