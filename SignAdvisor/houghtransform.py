import numpy as np
import math
import cv2
from matplotlib import pyplot as plt

def show_image(image):
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()

def rotate(vector, theta):

    rot_matrix = np.array(((np.cos(theta), -np.sin(theta)),
                           (np.sin(theta), np.cos(theta))))


    result = rot_matrix.dot(vector)
    return result

def calculate_barycenter(kp):
    x_barycenter = int(np.mean([i.pt[0] for i in kp]))
    y_baricenter = int(np.mean([i.pt[1] for i in kp]))
    return [x_barycenter, y_baricenter]

def hough_transform(good_matches, kp_train, kp_query, img_train):
    barycenter = calculate_barycenter(kp_query)
    scaling = np.arange(0.01, 1.01, 0.01)
    angle = np.arange(0, 361, 1)
    accumulator_array = np.zeros((img_train.shape[0], img_train.shape[1]), dtype=np.uint8)
    accumulator_array_2 = np.zeros((len(scaling), len(angle)), dtype=np.uint8)

    for m in good_matches:
        query_kp = kp_query[m.queryIdx]
        train_kp = kp_train[m.trainIdx]

        scale = round((train_kp.size/query_kp.size), 2)
        ref_vector = np.subtract(barycenter, query_kp.pt)
        vector_scaled = np.multiply(ref_vector, scale)
        vector_scaled = np.reshape(vector_scaled, (2,1))
        delta_phi = int(query_kp.angle - train_kp.angle)

        if delta_phi < 0:
            delta_phi = int(360 + delta_phi)
        rot_vector = rotate(vector_scaled, math.radians(delta_phi))
        rot_vector = rot_vector.reshape(2)

        estimate_x = int(round(train_kp.pt[0] + rot_vector[0]))
        estimate_y = int(round(train_kp.pt[1] + rot_vector[1]))

        y_estimate = [estimate_x, estimate_y]

        if 0 <= estimate_x < accumulator_array.shape[1] and 0 <= estimate_y < accumulator_array.shape[0]:
            accumulator_array[y_estimate[1], y_estimate[0]] += 1
            accumulator_array_2[np.where(scaling == scale), delta_phi] += 1

    y_target = np.unravel_index(accumulator_array.argmax(), accumulator_array.shape)
    dim = np.unravel_index(accumulator_array_2.argmax(), accumulator_array_2.shape)
    img_train[y_target[1], y_target[0]] = [0, 255, 0]
    cv2.circle(img_train, (y_target[1], y_target[0]), 10, (0, 0, 255), 2)
    show_image(img_train)
    return y_target, dim


