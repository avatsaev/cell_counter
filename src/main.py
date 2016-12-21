import preprocessor as preproc
import spot_detector
import cell_counter
import cv2
import argparse
import helper
import spot_reconstructor

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True,
    help = "Path to the image")




args = vars(ap.parse_args())


img = cv2.imread(args["image"])



#img = cv2.imread('contours_sample_2_raw.jpg')
# template = cv2.imread(args["template"], 0)


print("processing edges...")

#opening morph to remove points
opening_morph = preproc.opening_morph(cv2.medianBlur(img,5))

cv2.imwrite( "output/opening_morph.png", opening_morph );

histeq = preproc.histeq(opening_morph)

cv2.imwrite( "output/histeq.png", histeq );


# canny edge detection
edged_img = preproc.auto_canny_edge(histeq)
cv2.imwrite( "output/canny_edged.png", edged_img );


#dilate the edges for better pattern matching

dilated_img  = preproc.dilate(edged_img)
cv2.imwrite( "output/dilated.png", dilated_img );



###-START-CUT-1



(shapes, square_size, interspot_dist) = spot_detector.exeCalc(dilated_img)



#new_shapes = fill_in_missing_spots(shapes, square_size, interspot_dist)
#new_shapes = horizontal_spot_fill_in(shapes, square_size, interspot_dist)

# o_shape = find_origin_spot_shape(shapes, square_size, interspot_dist)
#
# if(not shape_exists(shapes, o_shape)):
#     shapes.insert(0, o_shape)


new_shapes = spot_reconstructor.get_missing_spots(shapes, square_size, interspot_dist, img)
squares = spot_detector.printSquare(new_shapes, img)


for i in range(len(new_shapes)):

    shape = new_shapes[i]

    #get region of interest (y1:y2, x1:x2)
    roi_img = img[shape[0][1]:shape[1][1],shape[0][0]:shape[1][0]]
    #count cells
    cells_n = cell_counter.count_cells(roi_img)
    cells_label = str(cells_n) +  " cell(s)"
    #draw text background
    cv2.rectangle(squares, (shape[0][0], shape[0][1]-40), (shape[1][0],  shape[0][1]), (0,255,0), cv2.FILLED)
    #draw label
    cv2.putText(squares, cells_label, (shape[0][0]+20, shape[0][1]-12), cv2.FONT_HERSHEY_SIMPLEX, 0.76, (0, 0, 0), 2)





cv2.imwrite("output/final.jpg", squares)


"INFO: test command: python main.py -i test_ressources/contours_sample_1_raw.jpg"


















##-END-CUT-1

# print "detecting spots..."
#
# res = cv2.matchTemplate(dilated_img,template,cv2.TM_CCOEFF_NORMED)
#

# threshold = 0.19
# loc = np.where( res >= threshold)

#print(loc)
# print "laying out spots.."
#
# spot_map_img = np.zeros(img.shape, np.uint8)
# template_w, template_h = template.shape[::-1]
#
# for pt in zip(*loc[::-1]):
#     cv2.rectangle(spot_map_img, pt, (pt[0] + template_w, pt[1] + template_h), (0,255,0), 3)
#
# cv2.imwrite('output/spot_map_img.png',spot_map_img)
#
#
# #this image contains only the detected areas of droplets
# spot_map_img = cv2.cvtColor(spot_map_img, cv2.COLOR_BGR2GRAY)
#
#
# cnts = cv2.findContours(spot_map_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
# cnts = cnts[0] if imutils.is_cv2() else cnts[1]
#
# sd = ShapeDetector()
# clean_spot_map_img = np.zeros(img.shape, np.uint8)
#
# # loop over the contours
# spot_n = 0
#
# aprox_contours = [cv2.approxPolyDP(c,epsilon=10.5,closed=True) for c in cnts]
#
# spot_coordinates = []
#
# for c in aprox_contours:
#   # compute the center of the contour, then detect the name of the
#   # shape using only the contour
#   M = cv2.moments(c)
#   cX = int((M["m10"] / M["m00"]))
#   cY = int((M["m01"] / M["m00"]))
#   shape = sd.detect(c)
#
#   if shape == "square" or shape=="rectangle":
#
#     c = c.astype("int")
#
#     #normalize irregular squares
#     c[0][0][0] = c[1][0][0]
#     c[0][0][1] = c[3][0][1]
#     c[1][0][1] = c[2][0][1]
#     c[2][0][0] = c[3][0][0]
#
#     spot_coords = [ c[0][0], c[1][0], c[2][0], c[3][0] ]
#
#     spot_coordinates.append(spot_coords)
#
#     cv2.drawContours(clean_spot_map_img, [c], -1, (0, 255, 0), 1)
#
#     spot_name = "Droplet {0}".format(spot_n)
#
#     #cv2.putText(clean_spot_map_img, spot_name, (cX-80, cY-70), cv2.FONT_HERSHEY_PLAIN, 1., (255, 255, 255), 1)
#
#     #cells_label = str(count_cells(img, spot_coords))+  " cell(s)"
#
#     #cv2.putText(clean_spot_map_img, cells_label, (cX-80, cY-45), cv2.FONT_HERSHEY_DUPLEX, 0.74, (255, 255, 255), 1)
#
#
#     #cv2.putText(clean_spot_map_img, number_of_cells, (cX-80, cY+35), cv2.FONT_HERSHEY_DUPLEX, 0.84, (255, 255, 255), 2)
#
#     spot_n += 1
#
# final_overlay_img = dynamic_blend( img,  clean_spot_map_img)
#
# cv2.imwrite('output/clean_spot_map_img.png',clean_spot_map_img)
# cv2.imwrite('output/final_overlay_img.png',final_overlay_img)
#
# print "PATTERN MATCHING DONE"
