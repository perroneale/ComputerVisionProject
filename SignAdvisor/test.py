import cv2
import numpy as np
from matplotlib import pyplot as plt

LINK = 'https://www.tripadvisor.it/Restaurant_Review-g187801-d10044689-Reviews-La_Prosciutteria_Bologna-Bologna_Province_of_Bologna_Emilia_Romagna.html'
TEST = 1

def show_image(image):
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()

def show_image_grayscale(image):
    plt.imshow(image, cmap='gray', vmin=0, vmax=255)
    plt.show()

def showKeyPoints(image, keyPoints):
    img_visualization_kp_query = cv2.drawKeypoints(image, keyPoints, None,
                                                   flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    plt.imshow(cv2.cvtColor(img_visualization_kp_query, cv2.COLOR_BGR2RGB))
    plt.show()

def find_query_in_train(good_matches, kp_query, kp_train, img_train, img_query):
    MIN_MATCH = 10
    projected_points = []
    if len(good_matches) > MIN_MATCH:
        query_pts = np.float32([kp_query[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
        train_pts = np.float32([kp_train[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)
        H, mask = cv2.findHomography(query_pts, train_pts, cv2.RANSAC, 5.0)
        print(H)
        h,w,_ = img_query.shape
        points_to_project = np.float32([[0,0],[w-1,0],[w-1,h-1],[0,h-1]]).reshape(-1,1,2)
        projected_points = cv2.perspectiveTransform(points_to_project,H)

        #img_train = cv2.polylines(img_train, [np.int32(projected_points)], True, [0,0,255],2,cv2.LINE_AA)
        return 0, projected_points
    else:
        print("Not enough matches found")
        return -1, projected_points


def add_rating(image, rating_value, sign_points):

    h_t,w_t, _ = image.shape
    img_rating = cv2.imread("../Rating/4.png")
    show_image(img_rating)
    h, w, _ = img_rating.shape
    rating_points = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]).reshape(-1, 1, 2)

    pt = cv2.getPerspectiveTransform(rating_points, sign_points)

    final_image = cv2.warpPerspective(img_rating, pt, (w_t, h_t))
    plt.imshow(final_image)
    plt.show()
    white_mask = np.ones((h,w),dtype=np.uint8)*255
    warped_mask = cv2.warpPerspective(white_mask, pt, (w_t, h_t))
    plt.imshow(warped_mask)
    plt.show()
    warped_mask = np.equal(warped_mask, np.array([0]))
    print(warped_mask)
    final_image[warped_mask] = image[warped_mask]
    show_image(final_image)
#1. leggo immagine scatta dall'utente, in questo caso la carico manualmente
#leggo anche l'immagine query

img_train = cv2.imread("../Sign_test_photo/test_prosciutteria.jpg")
img_query = cv2.imread("../Sign_ComputerVisionProject/la_prosciutteria2.png")

#img_query = cv2.cvtColor(img_q, cv2.COLOR_BGR2GRAY)
#img_train = cv2.cvtColor(img_t, cv2.COLOR_BGR2GRAY)

#img_train = cv2.GaussianBlur(img_train, (7,7), 1)
if TEST :
    show_image(img_train)
    show_image(img_query)
    #show_image_grayscale(img_train)
    #show_image_grayscale(img_query)
#istanzio SIFT

sift = cv2.xfeatures2d.SIFT_create()

#trovo i keypoints nell'immagine query

kp_query = sift.detect(img_query)
#visualizzare i keypoints

#trovo i keypoints nell'immagine di test
kp_train = sift.detect(img_train)
if TEST:
    showKeyPoints(img_query, kp_query)
    showKeyPoints(img_train, kp_train)

#ora devo calcolare i descriptor dei keypoints

kp_query, descriptor_query = sift.compute(img_query, kp_query)
kp_train, descriptor_train = sift.compute(img_train, kp_train)

#match descriptor

FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(descriptor_query,descriptor_train, k=2)
#remove bad matches
good_matches = []
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good_matches.append(m)

code, projected_points = find_query_in_train(good_matches, kp_query,kp_train,img_train,img_query)
if code == -1:
    print("Error!")
else:
    add_rating(img_train, 4, projected_points)
    #show_image(img_train)




