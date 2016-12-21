

def shape_exists(shapes, shape):
    exists = False
    tolerance = 10

    for sh in shapes:

        shape_p1 = shape[0]
        shape_p2 = shape[1]

        tmp_shape_p1 = sh[0]
        tmp_shape_p2 = sh[1]

        if ((shape_p1[0] >= tmp_shape_p1[0]-tolerance and shape_p1[0] <= tmp_shape_p2[0]+tolerance) and (shape_p1[1]>=tmp_shape_p1[1]-tolerance and shape_p1[1] <= tmp_shape_p2[1]+tolerance)) or ((shape_p2[0] <= tmp_shape_p2[0]+tolerance and shape_p2[0] >= tmp_shape_p1[0]-tolerance) and (shape_p2[1]<= tmp_shape_p2[1]+tolerance and shape_p2[1] >= tmp_shape_p1[1]-tolerance)):
            exists = True
            break

    return exists


def rectify_position(shapes, shape, tolerance=35):
    #find shape with same x positions+tolerance
    shape_x_rectified = False
    shape_y_rectified = False


    for tmp_shape in shapes:

        if shape[0][1] >= tmp_shape[0][1]-tolerance and shape[0][1] <= tmp_shape[0][1]+tolerance:
            shape[0][1] = tmp_shape[0][1]
            shape_y_rectified = True
            break


    for tmp_shape in shapes:

        if shape[0][0] >= tmp_shape[0][0]-tolerance and shape[0][0] <= tmp_shape[0][0]+tolerance:
            shape[0][0] = tmp_shape[0][0]
            shape_x_rectified = True
            break

    return shape






def count_lines_cols(shapes):

    lines = 0
    cols = 0

    new_shapes = shapes[:]


    for i in range(len(shapes)-1):

        shape = shapes[i]
        next_shape = shapes[i+1]

        vertical_distance =  next_shape[0][1] - shape[1][1]

        if vertical_distance<0:
            cols = cols+1
            if(lines==0):
                lines = i

    cols = cols+1
    lines = lines+1

    return (lines, cols)

def find_origin_spot_shape(shapes, square_size, interspot_dist):

    lines, cols = count_lines_cols(shapes)
    shape = shapes[0]
    x = shape[0][0]
    y = shape[0][1]

    while True:

        if  x - (interspot_dist+square_size) < 0 and y - (interspot_dist+square_size) < 0:
            break
        else:
            if x - (interspot_dist+square_size) >= 0:
                x = x-(interspot_dist+square_size)
            if y - (interspot_dist+square_size) >= 0:
                y = y-(interspot_dist+square_size)

    return [[x,y], [x+square_size, y+square_size]]


def get_missing_spots(shapes, square_size, interspot_dist, img):

    o_shape = find_origin_spot_shape(shapes, square_size, interspot_dist)
    missing_shapes = []

    tolerance = 10
    (img_rows, img_cols, c) = img.shape

    x = o_shape[0][0]
    y = o_shape[0][1]
    tmp_shape = [[x,y], [x+square_size, y+square_size]]

    while True:



        if(y+tolerance < img_rows and x+tolerance < img_cols):

            if(not shape_exists(shapes, tmp_shape)):

                tmp_shape = rectify_position(shapes, tmp_shape)

                tmp_shape[1][0] = tmp_shape[0][0]+square_size
                tmp_shape[1][1] = tmp_shape[0][1]+square_size

                missing_shapes.append(tmp_shape)

            if(x + (interspot_dist+square_size+tolerance) > img_cols):

                x = o_shape[0][0]
                y = y + (interspot_dist+square_size)

            else:
                x = x + (interspot_dist+square_size)



            tmp_shape = [[x,y], [x+square_size, y+square_size]]

        else:
            break

    return shapes+missing_shapes








def fill_in_missing_spots(shapes, square_size, interspot_dist):

    cols = 0
    interspot_dist_tolerance = 10
    new_shapes = shapes[:]

    for i in range(len(shapes)-1):

        shape = shapes[i]
        next_shape = shapes[i+1]

        vertical_distance =  next_shape[0][1] - shape[1][1]

        print vertical_distance

        if vertical_distance>interspot_dist+interspot_dist_tolerance:

            print "=="
            cols = cols + 1
            #
            missing_shape_pt_1 = [shape[0][0], shape[0][1]+square_size+interspot_dist]
            missing_shape_pt_2 = [shape[1][0], shape[1][1]+square_size+interspot_dist]
            missing_spot = [missing_shape_pt_1, missing_shape_pt_2]
            print missing_spot
            print shape
            new_shapes.insert(i,missing_spot)

    print cols+1
    return new_shapes
