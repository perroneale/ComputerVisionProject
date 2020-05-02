import mysql.connector
from mysql.connector import Error
from mysql.connector import errorcode
import numpy as np

config = {
    'host':'localhost',
    'user': 'root',
    'password': 'pass',
    'database':'signadvisor'}
try:
    connection = mysql.connector.connect(**config)
    if connection.is_connected():
        db_info = connection.get_server_info()
        print("Connected to MySQL Server version", db_info)
        cursor = connection.cursor()
        #eseguo le query
        cursor.execute("select database();")
        record = cursor.fetchone()
        cursor.close()
        print("You are connected to database: ",record)
except Error as e:
    print("Error while connecting to MySQL", e)

def query():
    cursor = connection.cursor()
    query = "SELECT * FROM sign"
    cursor.execute(query)
    for result in cursor:
        print("path {}, descriptor {}".format(result[1],result[5]))

def get_sign_name():
    cursor = connection.cursor()
    query = "SELECT photo FROM sign"
    cursor.execute(query)
    name = []
    for result in cursor:
        name.append(result[0])
    print(name)
    cursor.close()
    return name

def get_info_found_sign(name_sign):
    cursor = connection.cursor()
    query = ("SELECT review, url FROM sign WHERE photo =\""+name_sign+"\";")
    cursor.execute(query)
    result = []
    for review, url in cursor:
        print(review)
        print(url)
        result.append(review)
        result.append(url)
    return result

def insert_keypoints(keypoints, image_name):
    path = 'C:/Users/perro/Desktop/ComputerVisionProject/Sign_ComputerVisionProject/'
    complete_path = path+image_name
    update_query = ("UPDATE sign SET key_points = %s WHERE photo = %s;")
    cursor = connection.cursor()
    try:
        cursor.execute(update_query, (keypoints,complete_path))
    except mysql.connector.Error as err:
        print("Error code: ", err.errno)
    connection.commit()
    cursor.close()

def insert_descriptors(descriptor, image_name):
    path = 'C:/Users/perro/Desktop/ComputerVisionProject/Sign_ComputerVisionProject/'
    complete_path = path+image_name
    update_query = ("UPDATE sign SET descriptor = %s WHERE photo = %s;")
    cursor = connection.cursor()
    try:
        arr = np.ndarray.dumps(descriptor)
        print(arr)
        cursor.execute(update_query, (arr,complete_path))
    except mysql.connector.Error as err:
        print("Error code: ", err.errno)
    connection.commit()
    cursor.close()

def close_connection():
    connection.close()
    cursor.close()
    print("MySQL connection is closed")