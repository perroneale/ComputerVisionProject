import numpy as np
import cv2
from matplotlib import pyplot as plt
import math

def show_image_grayscale(image):
    plt.imshow(image, cmap='gray', vmin=0, vmax=255)
    plt.show()

def calculate_avereage(image):
    h,w = image.shape[:2]
    sum = 0
    for i in range(0,h-1):
        for j in range(0,w-1):
            sum = sum + image[i,j]
    mean = sum/(h*w)
    return mean

def zmNCC(img_model, img_target):
    h_m, w_m = img_model.shape[:2]
    h_t, w_t = img_target.shape[:2]
    img_model = cv2.resize(img_model, (w_t,h_t))
    #show_image_grayscale(img_model)
    print("Shape model ",img_model.shape)
    print("Shape train ",img_target.shape)
    av = calculate_avereage(img_model)
    print("Average my = ",av)
    average_t = calculate_avereage(img_target)
    numerator = 0
    denominator_m = 0
    denominator_t = 0
    n2 = np.sum(np.multiply((img_target - average_t),(img_model - av)))
    d2_m = np.sum(np.square(img_model - av))
    d2_t = np.sum(np.square(img_target -average_t))
    den_m = np.sqrt(np.multiply(d2_m,d2_t))
    score2 = n2/den_m
    for i in range(0,h_t-1):
        for j in range(0,w_t-1):
            numerator += (img_target[i,j] - av) * (img_model[i,j] - average_t)
            denominator_m += (img_target[i,j] - av)**2
            denominator_t += (img_model[i,j] - average_t)**2

    denominator_m = math.sqrt(denominator_m)
    denominator_t = math.sqrt(denominator_t)
    zmncc = numerator/(denominator_m*denominator_t)
    print("zmncc = ", zmncc)
    print("score 2 = ",score2)
    return zmncc

img_model = cv2.imread("../Sign_ComputerVisionProject/sonora.png",0)
img_target = cv2.imread("../Sign_ComputerVisionProject/sonora.png",0)

#score = zmNCC(img_model,img_target)
#print(score)


# h_m, w_m = img_model.shape[:2]
#     h_t, w_t = img_target.shape[:2]
#     img_model = cv2.resize(img_model, (w_t,h_t))
#     show_image_grayscale(img_model)
#     print("Shape model ",img_model.shape)
#     print("Shape train ",img_target.shape)
#     average_m = np.mean(img_model)
#     average_t = np.mean(img_target)
#     numerator = 0
#     denominator_m = 0
#     denominator_t = 0
#     for i in range(0,h_t-1):
#         for j in range(0,w_t-1):
#             numerator += (img_target[i,j] - average_m) * (img_model[i,j] - average_t)
#             denominator_m += (img_target[i,j] - average_m)**2
#             denominator_t += (img_model[i,j] - average_t)**2
#
#     denominator_m = math.sqrt(denominator_m)
#     denominator_t = math.sqrt(denominator_t)
#     zmncc = numerator/(denominator_m*denominator_t)