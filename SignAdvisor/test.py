import cv2
import numpy as np
from matplotlib import pyplot as plt
import webbrowser
import mySQLConnector
import generalizeHoughTransform as ght
import sys
import os

path = ""
TEST = 0
bottom_left_sign_position = []
height_sign = 0
width_sign = 0
PATH = "../Sign_ComputerVisionProject/"
projected_points = []
found = False
link = ""
found_name = ""

if len(sys.argv) < 2:
    print("Devi passare il path dell'immagine allo script")
    path = "../Sign_test_photo/test_prosciutteria.jpg"
    #sys.exit()
else:
    path = sys.argv[1]
    print(path)
    if os.path.isfile(path) == False:
        print("L'immagine non esiste")
        sys.exit()

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
    print(good_matches)
    if len(good_matches) >= MIN_MATCH:
        print("Keypoints")
        print(kp_train[0].angle)
        query_pts = np.float32([kp_query[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
        train_pts = np.float32([kp_train[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)
        print(query_pts)
        gradient_orientation_query = np.uint([kp_query[m.queryIdx].angle for m in good_matches]).reshape(-1,1)
        gradient_orientation_train = np.uint([kp_train[m.trainIdx].angle for m in good_matches]).reshape(-1,1)
        ght.create_R_table(query_pts,gradient_orientation_query,img_query.shape)
        r,scale = ght.online(gradient_orientation_train, train_pts,img_train.shape)
        copy = img_train.copy()
        cv2.circle(copy,(r[0],r[1]),10,(0,255,0),2,cv2.LINE_AA)
        h_q,w_q ,_ = img_query.shape
        h_q,w_q = h_q*scale,w_q*scale
        starting_point = (int(r[0] - w_q/2), int(r[1] - h_q/2))
        print(starting_point)
        finish_point = (int(r[0] + w_q/2), int(r[1] + h_q/2))
        print(finish_point)
        cv2.rectangle(copy,starting_point, finish_point, (0,0,255),1,cv2.LINE_AA)
        plt.imshow(cv2.cvtColor(copy, cv2.COLOR_BGR2RGB))
        plt.show()
        H, mask = cv2.findHomography(query_pts, train_pts, cv2.RANSAC, 5.0)
        h,w,_ = img_query.shape
        points_to_project = np.float32([[0,0],[w-1,0],[w-1,h-1],[0,h-1]]).reshape(-1,1,2)
        projected_points = cv2.perspectiveTransform(points_to_project,H)
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
    rating_value_string = str(rating_value)
    path_rating_image = "../Rating/"
    path_rating_image = path_rating_image + rating_value_string.replace(".", "_") + ".png"
    print(path_rating_image)
    img_rating = cv2.imread(path_rating_image)
    #img_rating = cv2.resize(img_rating, (width_sign, height_sign), interpolation=cv2.INTER_AREA)
    #show_image(img_rating)
    h, w, _ = img_rating.shape
    rating_points = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]).reshape(-1, 1, 2)
    pt = cv2.getPerspectiveTransform(rating_points, sign_points)
    final_image = cv2.warpPerspective(img_rating, pt, (w_t, h_t))
    white_mask = np.ones((h,w),dtype=np.uint8)*255
    warped_mask = cv2.warpPerspective(white_mask, pt, (w_t, h_t))
    warped_mask = np.equal(warped_mask, np.array([0]))
    final_image[warped_mask] = image[warped_mask]
    colour_to_mask = np.array([[255,255,255],
                      [104,104,104],
                      [112,112,112],
                      [120,120,120],
                      [128,128,128],
                      [135,135,135],
                      [143,143,143]], dtype=np.uint8)
    for i in colour_to_mask:
        test_mask = np.all(final_image == i , axis=-1)
        final_image[test_mask] = image[test_mask]

    bottom_left_corner_sign = sign_points[3][0]
    bottom_left_corner_sign = bottom_left_corner_sign.astype('int')
    print("Position before translation {} , {} ".format(bottom_left_corner_sign[0], bottom_left_corner_sign[1]))
    #p = (int(position[0] - width_sign/2), int(position[1] + height_sign/2))
    string_position = (bottom_left_corner_sign[0], int(bottom_left_corner_sign[1] + height_sign/2))
    print(string_position)
    print("Position after translation {}".format(string_position))
    font  = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(final_image, "Click Here for Info", string_position, font,0.5, (255,255,255), 1, cv2.LINE_AA)
    return final_image, height_sign, width_sign, bottom_left_corner_sign


def match_descriptor(descriptor_query, descriptor_train):
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(descriptor_query, descriptor_train, k=2)
    # remove bad matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)
    return good_matches

def open_browser(coordinate):
    h = range(bottom_left_sign_position[1], bottom_left_sign_position[1] + height_sign)
    w = range(bottom_left_sign_position[0], bottom_left_sign_position[0] + width_sign)
    if (coordinate[0][0] in w) & (coordinate[0][1] in h):
        webbrowser.open(link, new=2)

def capture_click(event, x, y, flags, params):
    global coordinates
    if event == cv2.EVENT_LBUTTONDOWN:
        coordinates = [(x,y)]
        open_browser(coordinates)
#1. leggo immagine scatta dall'utente, in questo caso la carico manualmente
#leggo anche l'immagine query

img_train = cv2.imread(path)
sign_name = mySQLConnector.get_sign_name()
# istanzio SIFT
final_image = []
sift = cv2.xfeatures2d.SIFT_create()
# trovo i keypoints nell'immagine di test
kp_train = sift.detect(img_train)
kp_train, descriptor_train = sift.compute(img_train, kp_train)
for name in sign_name:
    print("Current name {}".format(name))
    string = PATH+name
    print(string)
    img_query = cv2.imread(string)
    # trovo i keypoints nell'immagine query
    kp_query = sift.detect(img_query)
    showKeyPoints(img_query,kp_query)
    # ora devo calcolare i descriptor dei keypoints
    kp_query, descriptor_query = sift.compute(img_query, kp_query)
    # match descriptor
    good_matches = match_descriptor(descriptor_query,descriptor_train)
    code, projected_points = find_query_in_train(good_matches, kp_query, kp_train, img_train, img_query)
    print("Code ",code)
    if code == 0:
        found = True
        found_name = name
        break

if found:
    print(found_name)
    info = mySQLConnector.get_info_found_sign(str(found_name))
    link = info[1]
    final_image,height_sign,width_sign,bottom_left_sign_position = add_rating(img_train, info[0], projected_points)
    print(height_sign, width_sign)
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", capture_click)

    while True:
        cv2.imshow("image", final_image)

        key = cv2.waitKey(1)

        if key == ord("q"):
            break
    cv2.destroyAllWindows()
else:
    print("Ristorante non presente nel nostro elenco")
    sys.exit()



# temp = [{'point0':k.pt[0],'point1':k.pt[1],'size':k.size,'angle': k.angle, 'response': k.response, "octave":k.octave,
#         "id":k.class_id} for k in kp_query]
# json_str = json.dumps(temp)
# print(json_str)
# print(type(json_str))
# mySQLConnector.query()
# mySQLConnector.insert_keypoints(json_str, 'la_prosciutteria.png')
# mySQLConnector.query()
# mySQLConnector.close_connection()


# if TEST == 1:
#     showKeyPoints(img_query, kp_query)
#     showKeyPoints(img_train, kp_train)
# mySQLConnector.insert_descriptors(descriptor_query,"la_prosciutteria.png")
# mySQLConnector.query()
# mySQLConnector.close_connection()










