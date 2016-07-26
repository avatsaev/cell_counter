#! /usr/local/bin/python


import numpy as np
import cv2
import argparse


def detect_edges(img, min_val=5, max_val=22):

    min_val = float(min_val)
    max_val = float(max_val)

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('output/edges.png',gray_img)
    return cv2.Canny(gray_img, min_val,max_val)

def color_detection(img):

    lower = [32, 55, 35]
    upper = [55, 230, 55]

    # create NumPy arrays from the boundaries
    lower = np.array(lower, dtype = "uint8")
    upper = np.array(upper, dtype = "uint8")

    # find the colors within the specified boundaries and apply
    # the mask
    mask = cv2.inRange(img, lower, upper)
    output = cv2.bitwise_and(img, img, mask = mask)

    cv2.imwrite('output/green_cells.png',output)
    # show the images
    return output

def blob_detection(img):

    #
    # big_img = cv2.resize(img, (0,0), fx=3, fy=3)
    #
    # cv2.imwrite('output/big_img.png', big_img)

    kernel = np.ones((3,3 ), np.uint8)
    im  = cv2.dilate(img, kernel, iterations = 1)

    cv2.imwrite('output/dilated.png',im)
    # Set up the detector with default parameters.

    # im = cv2.adaptiveThreshold(im, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 101, 1)
    # cv2.imwrite('output/thresholded.png',im)

    (cnts, _) = cv2.findContours(im.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts_sorted = sorted(cnts, key = cv2.contourArea, reverse = True)[:10]
    return (cnts, cnts_sorted)

def count_cells(img):

    green_cells_img = color_detection(img)

    edged_img = detect_edges(green_cells_img)

    cnts, cnts_sorted = blob_detection(edged_img)

    return len(cnts)


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True,
    help = "Path to the query image")

args = vars(ap.parse_args())

img = cv2.imread(args["image"])

green_cells_img = color_detection(img)

edged_img = detect_edges(green_cells_img)

cnts, cnts_sorted = blob_detection(edged_img)
print len(cnts)
print len(cnts_sorted)
cell_count_label = "Cell count: {0}".format(len(cnts))
cv2.putText(img, cell_count_label, (5, 25), cv2.FONT_HERSHEY_DUPLEX, 0.60, (255, 255, 255), 1)
cv2.imwrite('output/final.png',img)

#
# screenCnt = None
#
#
# for c in cnts_sorted:
#
#     # approximate the contour
#     peri = cv2.arcLength(c, True)
#     approx = cv2.approxPolyDP(c, 0.02 * peri, True)
#
#     screenCnt = approx
#     print screenCnt
#     img = cv2.drawContours(img, [screenCnt], -1, (0, 255, 0), 1)
#
# cv2.imwrite('output/blobs.png',img)
