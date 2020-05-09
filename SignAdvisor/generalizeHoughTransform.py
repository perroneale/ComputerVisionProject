import numpy as np
import cv2
import math
r_table = {}
accumulator_array = []

def create_R_table(query_pts, query_orientation, image_shape):
    #image shape è un vettore con h,w
    query_pts = query_pts.astype(int)
    query_pts = query_pts.reshape(-1,2)
    #reference point

    y = [int(image_shape[1]/2), int(image_shape[0]/2)]
    print(image_shape)
    print(y)
    delta_phi = 1
    print("from ght")
    vector_list = []
    #print(train_orientation)
    for i in range(0, query_orientation.shape[0]):
        v_x = (y[0] - query_pts[i][0])**2
        v_y = (y[1] - query_pts[i][1])**2
        magnitude = math.sqrt(v_x + v_y)
        theta = math.cosh(v_x / magnitude)
        r = [magnitude, theta]
        gradient_index = str(query_orientation[i])
        if gradient_index in r_table:
            r_table[gradient_index].append(r)
        else:
            r_table[gradient_index] = [r]

    # print("dict")
    # for i,k in r_table.items():
    #     print(i,k)

def online(orientation_train, train_pts,train_shape, query_shape):
    print("shape", query_shape)
    h_q = query_shape[0]
    w_q = query_shape[1]
    h_t = train_shape[0]
    w_t = train_shape[1]
    x_scale = w_t/w_q
    y_scale = h_t/h_q
    print(x_scale, y_scale)
    train_pts = train_pts.astype(int).reshape(-1,2)
    print("shape train_pts ",train_pts.shape)
    print("shape train_pts ", orientation_train.shape)
    accumulator_array = np.zeros((train_shape[1], train_shape[0]))
    print(accumulator_array.shape)
    print(accumulator_array)
    for i in range(0, orientation_train.shape[0]):
        gradient_index = str(orientation_train[i])
        if gradient_index in r_table:
            r_vector = r_table[str(orientation_train[i])]
            for vector in r_vector:
                v_y = vector[0] * math.sin(vector[1])
                v_x = vector[0] * math.cos(vector[1])
                y_x = int( v_x - train_pts[i][0])
                y_y = int( v_y - train_pts[i][1])
                accumulator_array[y_x][y_y] = accumulator_array[y_x][y_y] + 1
        else:
            continue


    print(accumulator_array)

