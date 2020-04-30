import cv2
import numpy as np
from matplotlib import pyplot as plt
import webbrowser

LINK = 'https://www.tripadvisor.it/Restaurant_Review-g187801-d10044689-Reviews-La_Prosciutteria_Bologna-Bologna_Province_of_Bologna_Emilia_Romagna.html'
TEST = 0
position = []
height_sign = 0
width_sign = 0
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
        #print(H)
        h,w,_ = img_query.shape
        points_to_project = np.float32([[0,0],[w-1,0],[w-1,h-1],[0,h-1]]).reshape(-1,1,2)
        projected_points = cv2.perspectiveTransform(points_to_project,H)
        #print(projected_points)
        #img_train = cv2.polylines(img_train, [np.int32(projected_points)], True, [0,0,255],2,cv2.LINE_AA)
        return 0, projected_points
    else:
        print("Not enough matches found")
        return -1, projected_points


def add_rating(image, rating_value, sign_points):
    height_sign = int(sign_points[3][0][1] - sign_points[0][0][1])
    width_sign = int(sign_points[2][0][0] - sign_points[3][0][0])

    # sign_points[0] = sign_points[3]
    # sign_points[1] = sign_points[2]
    # sign_points[3][0][1] = sign_points[3][0][1] + height_sign
    # sign_points[2][0][1] = sign_points[2][0][1] + height_sign
    # print(sign_points)
    #test_img = image.copy()
    #test_img = cv2.polylines(test_img,[np.int32(sign_points)],True,[0,0,255],2,cv2.LINE_AA)
    #show_image(test_img)
    h_t,w_t, _ = image.shape
    img_rating = cv2.imread("../Rating/4.png")
    #img_rating = cv2.resize(img_rating, (width_sign, height_sign), interpolation=cv2.INTER_AREA)
    #show_image(img_rating)
    h, w, _ = img_rating.shape
    rating_points = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]).reshape(-1, 1, 2)
    pt = cv2.getPerspectiveTransform(rating_points, sign_points)
    final_image = cv2.warpPerspective(img_rating, pt, (w_t, h_t))
    # plt.imshow(final_image)
    # plt.show()
    white_mask = np.ones((h,w),dtype=np.uint8)*255
    warped_mask = cv2.warpPerspective(white_mask, pt, (w_t, h_t))
    # plt.imshow(warped_mask)
    # plt.show()
    warped_mask = np.equal(warped_mask, np.array([0]))
    final_image[warped_mask] = image[warped_mask]
    prova = np.array([[255,255,255],
                      [104,104,104],
                      [112,112,112],
                      [120,120,120],
                      [128,128,128],
                      [135,135,135],
                      [143,143,143]], dtype=np.uint8)
    for i in prova:
        test_mask = np.all(final_image == i , axis=-1)
        final_image[test_mask] = image[test_mask]

    position = sign_points[3][0]
    position = position.astype('int')
    print("Position before translation {} , {} ".format(position, position[1]))
    #p = (int(position[0] - width_sign/2), int(position[1] + height_sign/2))
    p = (position[0], int(position[1] + height_sign/2))
    print(p)

    print("Position after translation {}".format(position))
    font  = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(final_image, "Click Here for Info", p, font,0.5, (255,255,255), 1, cv2.LINE_AA)
    #show_image(final_image)
    return final_image, position, height_sign, width_sign
#1. leggo immagine scatta dall'utente, in questo caso la carico manualmente
#leggo anche l'immagine query

img_train = cv2.imread("../Sign_test_photo/test_prosciutteria.jpg")
img_query = cv2.imread("../Sign_ComputerVisionProject/la_prosciutteria2.png")
final_image = []
#img_query = cv2.cvtColor(img_q, cv2.COLOR_BGR2GRAY)
#img_train = cv2.cvtColor(img_t, cv2.COLOR_BGR2GRAY)

#img_train = cv2.GaussianBlur(img_train, (7,7), 1)
if TEST == 1:
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
if TEST == 1:
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
    final_image, position,height_sign,width_sign = add_rating(img_train, 4, projected_points)
    #show_image(img_train)

def open_browser(coordinate):
    h = range(position[1], position[1] + height_sign)
    w = range(position[0], position[0] + width_sign)
    if (coordinate[0][0] in w) & (coordinate[0][1] in h):
        webbrowser.open(LINK, new=2)



def capture_click(event, x, y, flags, params):
    global coordinates
    if event == cv2.EVENT_LBUTTONDOWN:
        coordinates = [(x,y)]
        open_browser(coordinates)


cv2.namedWindow("image")
cv2.setMouseCallback("image", capture_click)

while True:
    cv2.imshow("image", final_image)

    key = cv2.waitKey(1)

    if key == ord("q"):
        break

cv2.destroyAllWindows()




