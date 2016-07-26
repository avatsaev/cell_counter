#! /usr/local/bin/python

import numpy as np
import cv2
import argparse
import imutils
from pyimagesearch.shapedetector import ShapeDetector

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

def dilate(img, iterations=2):
    kernel = np.ones((5,5 ),np.uint8)
    return cv2.dilate(img,kernel,iterations = iterations)


#####################################################################################


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True,
    help = "Path to the query image")

ap.add_argument("-t", "--template", required = True,
    help = "Path to the template image")


ap.add_argument("-min", "--edge-min", required = False,
    help = "Min threshold for edges")


ap.add_argument("-max", "--edge-max", required = False,
    help = "Max threshold for edges")


args = vars(ap.parse_args())


img = cv2.imread(args["image"])


edge_min = args["edge_min"] if args["edge_min"] else 5
edge_max = args["edge_max"] if args["edge_max"] else 22


#img = cv2.imread('contours_sample_2_raw.jpg')
template = cv2.imread(args["template"], 0)


print("processing edges...")

#opening morph to remove points
opening_morph = opening_morph(cv2.medianBlur(img,5))

##cv2.imwrite( "output_pattern/opening_morph.png", opening_morph );

# canny edge detection
edged_img = detect_edges(opening_morph, edge_min, edge_max)
cv2.imwrite( "output_pattern/edged.png", edged_img );

#dilate the edges for better pattern matching

dilated_img  = dilate(edged_img)
cv2.imwrite( "output_pattern/dilated.png", dilated_img );

print "detecting spots..."

res = cv2.matchTemplate(dilated_img,template,cv2.TM_CCOEFF_NORMED)


threshold = 0.19
loc = np.where( res >= threshold)

print "laying out spots.."
spot_map_img = np.zeros(img.shape, np.uint8)
template_w, template_h = template.shape[::-1]

for pt in zip(*loc[::-1]):
    cv2.rectangle(spot_map_img, pt, (pt[0] + template_w, pt[1] + template_h), (0,255,0), 3)

##cv2.imwrite('output_pattern/res.png',img)



#this image contains only the detected areas of droplets
spot_map_img = cv2.cvtColor(spot_map_img, cv2.COLOR_BGR2GRAY)


cnts = cv2.findContours(spot_map_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

cnts = cnts[0] if imutils.is_cv2() else cnts[1]

sd = ShapeDetector()
clean_spot_map_img = np.zeros(img.shape, np.uint8)

# loop over the contours
spot_n = 0

aprox_contours = [cv2.approxPolyDP(c,epsilon=10.5,closed=True) for c in cnts]

spot_coordinates = []

for c in aprox_contours:
  # compute the center of the contour, then detect the name of the
  # shape using only the contour
  M = cv2.moments(c)
  cX = int((M["m10"] / M["m00"]))
  cY = int((M["m01"] / M["m00"]))
  shape = sd.detect(c)



  if shape == "square" or shape=="rectangle":

    c = c.astype("int")

    #normalize irregular squares
    c[0][0][0] = c[1][0][0]
    c[0][0][1] = c[3][0][1]
    c[1][0][1] = c[2][0][1]
    c[2][0][0] = c[3][0][0]

    spot_coordinates.append([ c[0][0], c[1][0], c[2][0], c[3][0] ])

    cv2.drawContours(clean_spot_map_img, [c], -1, (0, 255, 0), 1)

    spot_name = "Droplet {0}".format(spot_n)
    cv2.putText(clean_spot_map_img, spot_name, (cX-80, cY-50), cv2.FONT_HERSHEY_PLAIN, 1.7, (255, 255, 255), 1)

    cv2.putText(clean_spot_map_img, "Cell count:", (cX-80, cY-10), cv2.FONT_HERSHEY_DUPLEX, 0.74, (255, 255, 255), 1)
    cv2.putText(clean_spot_map_img, "0", (cX-80, cY+35), cv2.FONT_HERSHEY_DUPLEX, 0.84, (255, 255, 255), 2)
    spot_n += 1

final_overlay_img = dynamic_blend( img,  clean_spot_map_img)

cv2.imwrite('output_pattern/clean_spot_map_img.png',clean_spot_map_img)
cv2.imwrite('output_pattern/final_overlay_img.png',final_overlay_img)

print "PATTERN MATCHING DONE"
