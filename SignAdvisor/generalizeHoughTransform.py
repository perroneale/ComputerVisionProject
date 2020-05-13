import numpy as np
import cv2
import math

r_table = {}
accumulator_array = []
h_q = 0
w_q = 0

def calculatev_x_y(y_x, x_x):
    v_x = 0
    if y_x > x_x:
        v_x = y_x - x_x
    elif y_x < x_x:
        v_x = x_x - y_x
    else:
        v_x = 0
    return v_x

def create_R_table(query_pts, query_orientation, image_shape):
    #image shape è un vettore con h,w
    global h_q, w_q,r_table
    h_q, w_q, _ = image_shape
    query_pts = query_pts.astype(int)
    query_pts = query_pts.reshape(-1,2)
    #reference point
    y = [int(w_q/2), int(h_q/2)]
    print(image_shape)
    print(y)
    print("from ght")
    #print(train_orientation)
    for i in range(0, query_orientation.shape[0]):
        v_x = calculatev_x_y(y[0], query_pts[i][0])
        v_y = calculatev_x_y(y[1], query_pts[i][1])
        #uscirà v_x positiva o negativa in base alla posizione di x rispetto a y
        magnitude = math.sqrt(v_x**2 + v_y**2)
        theta = math.acos(v_x / magnitude)
        r = [magnitude, theta]
        gradient_index = str(query_orientation[i])
        if gradient_index in r_table:
            r_table[gradient_index].append(r)
        else:
            r_table[gradient_index] = [r]
    r_table = r_table
    #print("dict")
    # for i,k in r_table.items():
    #     print(i,k)

def online(orientation_train, train_pts,train_shape):
    # for i, k in r_table.items():
    #     print(i,k)
    # ratio_w_h_query = w_q/h_q
    h,w,_ = train_shape
    # print(ratio_w_h_query)
    min_dim = w_q
    #scaling = [1.0,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1]
    scaling = np.arange(0.01,1.01,0.01)
    print("scaling ",scaling)
    # max_dim = h_q
    # if w_q > h_q:
    #     min_dim = h_q
    #     max_dim = w_q
    train_pts = train_pts.astype(int).reshape(-1,2)
    print("shape train_pts ",train_pts.shape)
    print("shape train_pts ", orientation_train.shape)
    accumulator_array = np.zeros((h, w,len(scaling)), dtype=np.uint)
    print("Accumulator array shape ",accumulator_array.shape)
    # print(accumulator_array.shape)
    print("Orientation train shape {}  train pts shape{}".format(orientation_train.shape, train_pts.shape))
    for i in range(0, orientation_train.shape[0]):
        print(i)
        gradient_index = str(orientation_train[i])
        if gradient_index in r_table:
            print(gradient_index)
            r_vector = r_table[gradient_index]
            print(r_vector)
            for vector in r_vector:
                for scale in scaling:
                    print("vector",vector[0],vector[1])
                    print("Min dim ",min_dim)
                    v_y = scale*vector[0] * (math.sin(vector[1]))
                    v_x = scale*vector[0] * (math.cos(vector[1]))
                    print("v_x = {}, v_y = {}, train_pts = {}".format(v_x,v_y,train_pts[i]))
                    y_x = 0
                    y_y = 0
                    if math.degrees(vector[1]) >= 0 and math.degrees(vector[1]) <= 180:
                        print(math.degrees(vector[1]))
                        y_x = int(v_x + train_pts[i][0])
                        y_y = int(v_y + train_pts[i][1])
                        print("y_x = {}, y_y = {}".format(y_x, y_y))
                    else:
                        print(math.degrees(vector[1]))
                        y_x = int(train_pts[i][0] - v_x)
                        y_y = int(train_pts[i][1] - v_y)
                    print("y_x ={}  y_y = {}".format(y_x,y_y))
                    if y_x >= accumulator_array.shape[0] or y_y >= accumulator_array.shape[1]:
                        print("Errore Scaling {}".format(scale))
                    else:
                        accumulator_array[y_x][y_y][np.where(scaling == scale)] = accumulator_array[y_x][y_y][np.where(scaling == scale)] + 1
                # prev_min_dim = min_dim
                # min_dim = min_dim - 1
                # prev_max_dim = max_dim
                # max_dim = ratio_w_h_query*min_dim
    y_target = np.unravel_index(accumulator_array.argmax(), accumulator_array.shape)
    print("Maximum y = {}, scale factor = {}".format(y_target, scaling[y_target[2]]))
    return y_target, scaling[y_target[2]]
