import cv2
import numpy as np
from matplotlib import pyplot as plt
import webbrowser
import mySQLConnector
import sys
import os
import pyautogui
import math
import houghtransform
import imutils

path = ""
img_train = []
TEST = 0
bottom_left_sign_position = []
height_sign = 0
width_sign = 0
PATH = "../Sign_ComputerVisionProject/"
projected_points = []
found = False
link = ""
found_name = ""
img_query = []

if len(sys.argv) < 2:
    print("Devi passare il nome, inclusa l'estensione dell'immagine, allo script")
    #path = "../Sign_test_photo/rusticone.jpg"
    sys.exit()
else:
    path = "../Sign_test_photo/"+sys.argv[1]
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

def zmNCC(img_model, img_target):
    h_t, w_t = img_target.shape[:2]
    #print(w_t,h_t)
    img_model = cv2.resize(img_model, (w_t,h_t))
    average_m = np.mean(img_model)
    average_t = np.mean(img_target)
    numerator = np.sum(np.multiply((img_target - average_t), (img_model - average_m)))
    denominator_m = np.sum(np.square(img_model - average_m))
    denominator_t = np.sum(np.square(img_target - average_t))
    den_m = np.sqrt(np.multiply(denominator_m, denominator_t))
    zmncc = numerator / den_m
    return zmncc

def ssd(img_model, img_target):
    h_t, w_t = img_target.shape[:2]
    #print(w_t,h_t)
    img_model = cv2.resize(img_model, (w_t,h_t))
    diff = np.subtract((img_target), (img_model))
    square = np.square(diff)
    SSD = np.sum(square)
    print(SSD)
    return SSD

#Determino la posizione della model image all'interno della target image
#Per determinare la posizione, partendo dalle corrispondenze presenti in good_matches
#tra model e target image, devo calcolare una trasformazione che mi permetta di proiettare i punti
#da model a target image.
def estimate_position(good_matches, kp_query, kp_train, img_train_bw, img_query_bw,img_train):
    MIN_MATCH = 15
    projected_points = []
    scaling = np.arange(0.01,1.01,0.01)
    if len(good_matches) >= MIN_MATCH:
        query_pts = np.float32([kp_query[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
        train_pts = np.float32([kp_train[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)

        # gradient_orientation_query = np.uint([kp_query[m.queryIdx].angle for m in good_matches]).reshape(-1,1)
        # gradient_orientation_train = np.uint([kp_train[m.trainIdx].angle for m in good_matches]).reshape(-1,1)
        # scale_query = np.uint([kp_query[m.queryIdx].size for m in good_matches])
        # scale_train = np.uint([kp_train[m.trainIdx].size for m in good_matches])
        y_target,dim = houghtransform.hough_transform(good_matches, kp_train, kp_query, img_train)

        h_q, w_q = img_query_bw.shape
        scale = scaling[dim[0]]
        print(scale)
        rotation =  - dim[1]
        rot_img_train = imutils.rotate(img_train_bw,rotation)
        h_q, w_q = h_q * scale, w_q * scale
        if y_target[1] - w_q/2 <= 0 or y_target[0] - h_q/2 <= 0:
            print("Second check fails")
            return -1, projected_points
        else:
            #copy = img_train.copy()
            # cv2.circle(copy,(y_target[0],y_target[1]),10,(0,255,0),2,cv2.LINE_AA)

            starting_point = (int(y_target[1] - w_q/2), int(y_target[0] - h_q/2))
            roi = rot_img_train[starting_point[1]:starting_point[1]+int(h_q),starting_point[0]:starting_point[0]+int(w_q)]
            #show_image_grayscale(roi)
            score = zmNCC(img_query_bw, roi)
            print("ZMNCC = ",score)
            if 0.5 <= score <= 1:
                # cv2.rectangle(copy, starting_point, finish_point, (0, 0, 255), 1, cv2.LINE_AA)
                # plt.imshow(cv2.cvtColor(copy, cv2.COLOR_BGR2RGB))
                # plt.show()
                # per stimare l'omografia utilizzo l'algoritmo RANSAC che permette di considerare solo i match corretti e non
                # quelli alterati dal rumore.
                H, mask = cv2.findHomography(query_pts, train_pts, cv2.RANSAC, 5.0)
                h, w = img_query_bw.shape
                # proietto un rettangolo, con le dimensioni della model image nella target image utilizzando l'omografia
                points_to_project = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]).reshape(-1, 1, 2)
                projected_points = cv2.perspectiveTransform(points_to_project, H)
                img_train = cv2.polylines(img_train, [np.int32(projected_points)],True, (0,0,255),3,cv2.LINE_AA)
                #show_image(img_train)
                return 0, projected_points
            else:
                print("Second check fails")
                return -1, projected_points
    else:
        print("Not enough matches found")
        return -1, projected_points

def add_rating(image_train, rating_value, sign_points,window_w, window_h):
    height_sign = int(sign_points[3][0][1] - sign_points[0][0][1])
    width_sign = int(sign_points[2][0][0] - sign_points[3][0][0])
    h_t,w_t, _ = image_train.shape
    rating_value_string = str(rating_value)
    path_rating_image = "../Rating/"
    path_rating_image = path_rating_image + rating_value_string.replace(".", "_") + ".png"
    img_rating = cv2.imread(path_rating_image)

    h, w, _ = img_rating.shape
    rating_points = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]).reshape(-1, 1, 2)

    #calcolo omografia per proiettare l'immagine contenente il rating sull'insegna
    PT = cv2.getPerspectiveTransform(rating_points, sign_points)
    #effettuo il warping dell'immagine
    final_image = cv2.warpPerspective(img_rating, PT, (w_t, h_t))
    #creo una white mask per andare a capire quali sono i pixel neri da sostituire con i pixel dell'immagine target
    white_mask = np.ones((h,w),dtype=np.uint8)*255
    warped_mask = cv2.warpPerspective(white_mask, PT, (w_t, h_t))
    warped_mask = np.equal(warped_mask, np.array([0]))
    #sostituisco i pixel neri
    final_image[warped_mask] = image_train[warped_mask]
    #array contenente i colori presenti nello sfondo dell'immagine di rating
    colour_to_mask = np.array([[255,255,255],
                      [104,104,104],
                      [112,112,112],
                      [120,120,120],
                      [128,128,128],
                      [135,135,135],
                      [143,143,143]], dtype=np.uint8)
    #mantengo solo le stelle andando ad eliminare lo sfondo
    for i in colour_to_mask:
        test_mask = np.all(final_image == i , axis=-1)
        final_image[test_mask] = image_train[test_mask]
    #aggiungo il testo cliccabile per ottenere aorire il browser con l'indirizzo tripadvisor
    bottom_left_corner_sign = sign_points[3][0]
    bottom_left_corner_sign = bottom_left_corner_sign.astype('int')
    string_position = (bottom_left_corner_sign[0], int(bottom_left_corner_sign[1] + height_sign/2 + 10))

    font  = cv2.FONT_HERSHEY_DUPLEX
    fontscale = 2
    cv2.putText(final_image, "More Info", string_position, font,fontscale, (0,255,0), 2, cv2.LINE_AA)
    return final_image, height_sign, width_sign, bottom_left_corner_sign

#il match dei descriptor viene eseguito tramite kd_tree
#Fast Library for Approximate Nearest Neighbors
def match_descriptor(descriptor_query, descriptor_train):
    FLANN_INDEX_KDTREE = 1
    #Per selezionare l'argoritmo da utilizzare
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    #It specifies the number of times the trees in the index should be recursively traversed.
    #Higher values gives better precision, but also takes more time.
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

def compute_kp_descr(image,sift):
    kp = sift.detect(image)
    kp, descriptor = sift.compute(image, kp)
    return kp, descriptor

#Carico immagine passata come argomento dall'utente
img_train = cv2.imread(path)
show_image(img_train)
img_train_bw = cv2.cvtColor(img_train, cv2.COLOR_BGR2GRAY)
#Estraggo dal database i nomi delle immagini delle insegne
sign_name = mySQLConnector.get_sign_name()
final_image = []
# istanzio SIFT
sift = cv2.xfeatures2d.SIFT_create()
# trovo i keypoints e descriptors dell'immagine di test
kp_train, descriptor_train = compute_kp_descr(img_train_bw, sift)

# showKeyPoints(img_train_bw,kp_train)
#per tutte le possibili insegne eseguo il math
for name in sign_name:
    print("Current restaurant {}".format(name))
    string = PATH+name
    img_query = cv2.imread(string)
    img_query_bw = cv2.cvtColor(img_query, cv2.COLOR_BGR2GRAY)
    # trovo i keypoints e descriptors dell'immagine query

    kp_query, descriptor_query = compute_kp_descr(img_query_bw, sift)
    # eseguo il match dei descriptor
    good_matches = match_descriptor(descriptor_query,descriptor_train)
    code, projected_points = estimate_position(good_matches, kp_query, kp_train, img_train_bw, img_query_bw,img_train)
    if code == 0:
        print("La foto scattata corrisponde al ristorante {}".format(name))
        found = True
        found_name = name
        break
    else:
        print("La foto scattata non corrisponde al ristorante {}".format(name))

#se ho trovato un'insegna che matcha con la mia target image estraggo dal database i dati relativi
#al rating ed il link tripadvisor
if found:
    info = mySQLConnector.get_info_found_sign(str(found_name))
    link = info[1]

    cv2.namedWindow("image",cv2.WINDOW_KEEPRATIO)
    w_size, h_size = pyautogui.size()
    final_image,height_sign,width_sign,bottom_left_sign_position = add_rating(img_train, info[0], projected_points,int(w_size/2),int(h_size/2))
    h_f, w_f ,_ = final_image.shape
    ratio = w_f / h_f

    final_image_resized = cv2.resize(final_image,(int(w_size/2),int((w_size/2))))
    ratio_h = final_image.shape[0]/final_image_resized.shape[0]
    ratio_w = final_image.shape[1] / final_image_resized.shape[1]
    bottom_left_sign_position[0] = int(bottom_left_sign_position[0]/ratio_w)
    bottom_left_sign_position[1] = int(bottom_left_sign_position[1]/ratio_h)
    cv2.rectangle(final_image_resized,(bottom_left_sign_position[0], bottom_left_sign_position[1]),(bottom_left_sign_position[0] + width_sign, bottom_left_sign_position[1] + height_sign) , (255, 0, 0), 1)
    cv2.moveWindow("image",0,0)
    #cv2.resizeWindow("image", int(w_size/2),int(h_size/2))
    cv2.setMouseCallback("image", capture_click)

    while True:
        cv2.imshow("image", final_image_resized)

        key = cv2.waitKey(1)

        if key == ord("q"):
            break
    cv2.destroyAllWindows()
else:
    print("Ristorante non presente nel nostro elenco")
    sys.exit()











