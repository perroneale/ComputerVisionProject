import numpy as np
import math
import time

accumulator_array = []

def calculatev_x_y(y_x, x_x):
    if y_x > x_x:
        v_x = y_x - x_x
    elif y_x < x_x:
        v_x = x_x - y_x
    else:
        v_x = 0
    return v_x

#off-line phase, creo la r_Table
def create_R_table(query_pts, query_orientation, image_shape):
    print("Inizio creazione R_table")
    r_table = {}
    h_q, w_q = image_shape
    query_pts = query_pts.astype(int)
    query_pts = query_pts.reshape(-1,2)
    #considero come referance point in centro dell'immagine
    y = [int(w_q/2), int(h_q/2)]
    #calcolo modulo e orientazione (rispetto all'asse j) dei vettori r
    for i in range(0, query_orientation.shape[0]):
        v_x = calculatev_x_y(y[0], query_pts[i][0])
        v_y = calculatev_x_y(y[1], query_pts[i][1])
        magnitude = math.sqrt(v_x**2 + v_y**2)
        theta = math.acos(v_x / magnitude)
        r = [magnitude, theta]
        gradient_index = str(query_orientation[i])
        #inserisco il valore di r nella r_table in corrispondenza dell'orientazione del gradiente
        if gradient_index in r_table:
            r_table[gradient_index].append(r)
        else:
            r_table[gradient_index] = [r]
    print("Terminata creazione R_table")
    print(r_table)
    return r_table



#online phase, istanzio un accumulator array a 3 dimensioni, i,j,scaling
def online(orientation_train, train_pts,train_shape, r_table):
    print(orientation_train)
    start_time = time.time()
    print("Inizio online phase GHT")
    h,w= train_shape
    #vettore con tutti i possibili valori di scala da 0.01 a 1 con passo di 0.01
    #effettuando degli esperimenti (impostando un passo di 0.01) il tempo impiegato per eseguire la fase online
    #aumentava considerevolmente, ottenendo una posizione di y più precisa di un solo pixel sulla coordinata x.
    scaling = np.arange(0.01,1.01,0.01)
    #angle = np.arange(0,361,1)
    print(scaling)
    train_pts = train_pts.astype(int).reshape(-1,2)
    accumulator_array = np.zeros((h, w,len(scaling)), dtype=np.uint)
    len_r_tabel = len(r_table)
    print(len_r_tabel)
    accumulator = 0
    for i in range(0, orientation_train.shape[0]):
        gradient_index = str(orientation_train[i])
        if gradient_index in r_table:
            print("Gradient_index ",gradient_index)
            #estraggo i vettori associati al gradiente del keypoints considerato
            r_vector = r_table[gradient_index]
            print(r_vector)
            #per ogni vettore calcolo le coordinate di y, considerando tutti i possibili valori di scala
            for vector in r_vector:
                for scale in scaling:
                    v_y = scale*vector[0] * (math.sin(vector[1]))
                    v_x = scale*vector[0] * (math.cos(vector[1]))
                    #print("v_x = {}, v_y = {}, train_pts = {}".format(v_x,v_y,train_pts[i]))
                    if math.degrees(vector[1]) >= 0 and math.degrees(vector[1]) <= 180:
                        #print(math.degrees(vector[1]))
                        y_x = int(v_x + train_pts[i][0])
                        y_y = int(v_y + train_pts[i][1])
                        #print("y_x = {}, y_y = {}".format(y_x, y_y))
                    else:
                        #print(math.degrees(vector[1]))
                        y_x = int(train_pts[i][0] - v_x)
                        y_y = int(train_pts[i][1] - v_y)
                    #print("y_x ={}  y_y = {}".format(y_x,y_y))
                    #se il fattore di scala non è corretto si potrbbero ottenere coordinate di y che non rientrano
                    #nella shape dell'immagine, quindi scarto questi valori
                    if y_x >= accumulator_array.shape[0] or y_y >= accumulator_array.shape[1]:
                        print("Errore Scaling {}".format(scale))
                    else:
                        #effettuo la votazione
                       accumulator_array[y_x][y_y][np.where(scaling == scale)] = accumulator_array[y_x][y_y][np.where(scaling == scale)] + 1
        else:
            accumulator += 1
    #ottengo le coordinate ed il fattore di scala della posizione nell'accumulator array con il maggior numero di voti
    print(accumulator)
    y_target = np.unravel_index(accumulator_array.argmax(), accumulator_array.shape)
    partial_candidate = []
    if accumulator >= len_r_tabel:
        code = -1
    else:
        value = accumulator_array[y_target]
        candidate = np.argwhere(accumulator_array == value)
        for w in candidate:
            if w[2] > 0:
                partial_candidate.append(w)
        code = 0
        print(partial_candidate)
    partial_candidate = np.array(partial_candidate, dtype=np.uint)
    print("Maximum y = {}, scale factor = {}, value = {}".format(y_target, scaling[y_target[2]], accumulator_array[y_target]))
    elapsed_time = time.time() - start_time
    print("Tempo impiegato fase online ",elapsed_time)
    print("Fine online phase GHT")
    return y_target, scaling[y_target[-1]],code, partial_candidate
