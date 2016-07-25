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

def bitwise_blend(img1, img2):

    rows, cols = img2.shape
    roi = img1[0:rows, 0:cols ]

    mask_inv = cv2.bitwise_not(img2)

    img1_bg = cv2.bitwise_and(roi, roi, mask = mask_inv)
    img2_fg = cv2.bitwise_and(img2, img2, mask = img2)

    return cv2.add(img1_bg, img2_fg)

def blob_detection(im):


    # Set up the detector with default parameters.
    detector = cv2.SimpleBlobDetector()

    # Detect blobs.
    keypoints = detector.detect(im)

    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0,0,0))
    return im_with_keypoints



print("processing edges...")

# # construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-q", "--query", required = True,
#     help = "Path to the query image")
# args = vars(ap.parse_args())
#
# # load the query image, compute the ratio of the old height
# # to the new height, clone it, and resize it
# image = cv2.imread(args["query"])


img = cv2.imread('contours_sample_1_raw.jpg')


ratio = 1



img = cv2.medianBlur(img,5)
#opening morph to remove points
kernel = np.ones((5,5 ),np.uint8)
opening_morph = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
cv2.imwrite( "output/opening_morph.png", opening_morph );



gray_img = cv2.cvtColor(opening_morph, cv2.COLOR_BGR2GRAY)
edged_img = cv2.Canny(gray_img, 5, 22)

cv2.imwrite( "output/edged.png", edged_img );




cnts = cv2.findContours(edged_img.copy(), cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
sd = ShapeDetector()

# loop over the contours
for c in cnts:
    # compute the center of the contour, then detect the name of the
    # shape using only the contour
    M = cv2.moments(c)
    if M["m00"] != 0:
        cX = int((M["m10"] / M["m00"]) )
        cY = int((M["m01"] / M["m00"]))
        shape = sd.detect(c)

        # multiply the contour (x, y)-coordinates by the resize ratio,
        # then draw the contours and the name of the shape on the img
        c = c.astype("float")
        c *= ratio
        c = c.astype("int")
        if shape=="rectangle":
            cv2.drawContours(img, [c], -1, (0, 255, 0), 2)

            cv2.putText(img, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 2)



cv2.imwrite("output/shapes.png", img)


# blobs_img = blob_detection(edged_img)
# cv2.imwrite("output/blobs.png", blobs_img)



#
# cv2.imwrite( "output/edges.png", edged_img );
#
#
# kernel = np.ones((6,1 ),np.uint8)
# eroded_img_v = cv2.erode(edged_img,kernel,iterations = 1)
#
# kernel = np.ones((1,6 ),np.uint8)
# eroded_img_h = cv2.erode(edged_img,kernel,iterations = 1)
#
#
# blended_erode = bitwise_blend(eroded_img_v, eroded_img_h)
#
#
# cv2.imwrite( "output/blended_erode.png", blended_erode);
# #
# # cv2.imwrite( "output/eroded_blend.png", blended_erode );
#
# kernel = np.ones((5,5 ),np.uint8)
# dilated_img  = cv2.dilate(blended_erode,kernel,iterations = 1)
#
#
#
# #
# # cv2.imwrite( "output/eroded_blend.png", blended_erode );
#
#
#
#
#
# cv2.imwrite( "output/dilated_blend2.png", dilated_img);
#
#
#
# #
# #
# # (cnts, _) = cv2.findContours(dilated_img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# #
# # cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:10]
# # screenCnt = None
# #
# #
# # for c in cnts:
# #     # approximate the contour
# #     peri = cv2.arcLength(c, True)
# #     approx = cv2.approxPolyDP(c, 0.02 * peri, True)
# #
# #     # if our approximated contour has four points, then
# #     # we can assume that we have found our screen
# #     print approx
# #     # if len(approx) == 4:
# #     print "SHAPE DETECTED!!"
# #     screenCnt = approx
# #
# #     cv2.drawContours(img, [screenCnt], -1, (0, 255, 0), 3)
# #
# # cv2.imwrite( "output/dilated_blend_shaped.png", img);
#
#
# cv2.waitKey(0)
