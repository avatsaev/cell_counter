#! /usr/local/bin/python

import numpy as np
import cv2
import argparse

def count_cells( spot_file):
	# Load the spot 
	img = cv2.imread(spot_file,1)
	
	# Separate the RGB channels
	RGB = cv2.split(img)
	
	# Extract the Green channel
	G = RGB[1]
	
	# Binarise the Green channel to separate cells from background
	ret1,th1 = cv2.threshold(G,64,255,cv2.THRESH_BINARY)
	#th1 = np.uint8(th1)

	# You need to choose 4 or 8 for connectivity type
	connectivity = 8  
	# Perform the operation
	num_labels,labels,stats,centroids = cv2.connectedComponentsWithStats(th1, connectivity, cv2.CV_32S)
	
	# Display process
	# cv2.imshow('color',img)
	# cv2.imshow('Green',G)
	# cv2.imshow('bw',th1)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()
	
	return num_labels -1

print count_cells("sample_4.jpg")
# connectedComponentsWithStats
# http://stackoverflow.com/questions/35854197/how-to-use-opencvs-connected-components-with-stats-in-python