import numpy as np
import math
import cv2
from matplotlib import pyplot as plt

def show_image(image):
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()

def calculatev_x_y(y_x, x_x):
    if y_x > x_x:
        v_x = y_x - x_x
    elif y_x < x_x:
        v_x = x_x - y_x
    else:
        v_x = 0
    return v_x

def off_line(query_pts, orientation, scale,img_query):
    orientation = orientation.astype(np.int)
    set_F = []
    h,w,_ = img_query.shape
    x_query_pts = []
    y_query_pts = []
    for q in range(0,query_pts.shape[0]):
        x_query_pts.append(query_pts[q][0][0])
        y_query_pts.append(query_pts[q][0][1])
    print(x_query_pts)
    print(y_query_pts)
    x_barycenter = int(w/2)
    y_baricenter = int(h/2)
    # x_barycenter = int(np.mean(x_query_pts))
    # y_baricenter = int(np.mean(y_query_pts))
    img_query[y_baricenter,x_barycenter] = [0,255,0]
    show_image(img_query)
    for i in range(0,orientation.shape[0]):
        # f = np.array([x_query_pts[i],y_query_pts[i]])
        # magnitude = np.linalg.norm(b-f)
        v_x = calculatev_x_y(x_barycenter, x_query_pts[i])
        v_y = calculatev_x_y(y_baricenter, y_query_pts[i])
        magnitude = math.sqrt(v_x**2 + v_y**2)
        theta = math.acos(v_x / magnitude)
        f = [x_query_pts[i],y_query_pts[i],orientation[i][0],scale[i],magnitude,theta]
        set_F.append(f)
        cv2.line(img_query,(x_query_pts[i], y_query_pts[i]),(x_barycenter,y_baricenter),(255,0,0),1)
        #inserisco il valore di r nella r_table in corrispondenza dell'orientazione del gradiente
    show_image(img_query)
    return set_F

def on_line_phase(train_pts, orientation_train, scale_train, set_F,img_train):
    h,w,_ = img_train.shape
    orientation_train = orientation_train.astype(np.int)
    print(set_F)
    scaling = np.arange(0.01, 1.01, 0.01)
    angle = np.arange(0, 361, 1)
    accumulator_array = np.zeros((h, w, len(scaling), len(angle)), dtype=np.uint8)
    for i in range(0, orientation_train.shape[0]):
        for scale in scaling:
            print("START")
            print("gradient train ",orientation_train[i][0])
            r_vector = set_F[i]
            print("current ",r_vector)
            print("gradient model ", r_vector[2])
            theta = int(r_vector[2] - orientation_train[i][0])
            print(theta)
            if theta < 0 :
                theta = int(360 + theta)
            print(theta)
            # estraggo i vettori associati al gradiente del keypoints considerato
            v_y = scale * r_vector[4] * (math.sin(theta))
            v_x = scale * r_vector[4] * (math.cos(theta))
            print(v_y,v_x)
            if theta >= 0 and theta <= 180:
                # print(math.degrees(vector[1]))
                y_x = int(v_x + train_pts[i][0][0])
                y_y = int(v_y + train_pts[i][0][1])
            else:
                # print(math.degrees(vector[1]))
                y_x = int(train_pts[i][0][0] - v_x)
                y_y = int(train_pts[i][0][1] - v_y)
            if y_x >= accumulator_array.shape[0] or y_y >= accumulator_array.shape[1]:
                print("Errore Scaling {}".format(scale))
            else:
                # effettuo la votazione
                print("voto at scaling ",scale)
                accumulator_array[y_x, y_y, np.where(scaling == scale), int(theta)] = accumulator_array[y_x, y_y, np.where(scaling == scale), int(theta)] +  1

    print("fine")
    y_target = np.unravel_index(accumulator_array.argmax(), accumulator_array.shape)
    print(y_target)
    img_train[y_target[1],y_target[0]] = [0,255,0]
    cv2.circle(img_train,(y_target[0],y_target[1]),5,(0,0,255),1)
    show_image(img_train)
    #value = accumulator_array[y_target[0],y_target[1],y_target[2],y_target[3]]
    # print("value =",value)
    # candidate = np.argwhere(accumulator_array == value)
    #
    # print(candidate)
    # candidate = np.array(candidate, dtype=np.uint)
    # for p in candidate:
    #     cp = img_train.copy()
    #     for t in train_pts:
    #         cv2.line(cp, (t[0], t[1]), (p[0], p[1]), (255, 0, 0), 1)
    #     plt.imshow(cv2.cvtColor(cp, cv2.COLOR_BGR2RGB))
    #     plt.show()
    return y_target
