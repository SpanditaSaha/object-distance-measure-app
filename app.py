import os
from flask import Flask, render_template, Response
import cv2 as cv
import numpy as np

app = Flask(__name__)
env_config = os.getenv("PROD_APP_SETTINGS", "config.DevelopmentConfig")
app.config.from_object(env_config)

# Distance constants
KNOWN_DISTANCE = 45  # INCHES
PERSON_WIDTH = 16  # INCHES
MOBILE_WIDTH = 3.0  # INCHES

# Object detector constant
CONFIDENCE_THRESHOLD = 0.4
NMS_THRESHOLD = 0.3

# Colors for object detected
COLORS = [(255, 0, 0), (255, 0, 255), (0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)

# Defining fonts
FONTS = cv.FONT_HERSHEY_COMPLEX

# Getting class names from classes.txt file
class_names = []
with open("classes.txt", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]

# Setting up OpenCV net
yoloNet = cv.dnn.readNet('yolov4-tiny.weights', 'yolov4-tiny.cfg')

yoloNet.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
yoloNet.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)

model = cv.dnn_DetectionModel(yoloNet)
model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)

# Object detector function
def object_detector(image):
    classes, scores, boxes = model.detect(image, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    # Creating an empty list to add object data
    data_list = []
    for (classid, score, box) in zip(classes, scores, boxes):
        # Define color of each object based on its class id
        color = COLORS[int(classid) % len(COLORS)]

        label = "%s : %f" % (class_names[classid], score)

        # Draw rectangle and label on object
        cv.rectangle(image, box, color, 2)
        cv.putText(image, label, (box[0], box[1]-14), FONTS, 0.5, color, 2)

        # Getting the data
        # 1: class name, 2: object width in pixels, 3: position where text needs to be drawn (distance)
        if classid == 0:  # Person class id
            data_list.append([class_names[classid], box[2], (box[0], box[1]-2)])
        elif classid == 67:  # Cell phone class id
            data_list.append([class_names[classid], box[2], (box[0], box[1]-2)])
        # If you want to include more classes, add more 'elif' statements here
        elif classid == 39:
            data_list.append([class_names[classid], box[2], (box[0], box[1]-2)])
        elif classid == 41:
            data_list.append([class_names[classid], box[2], (box[0], box[1]-2)])
        elif classid == 42:
            data_list.append([class_names[classid], box[2], (box[0], box[1]-2)])
        elif classid == 43:
            data_list.append([class_names[classid], box[2], (box[0], box[1]-2)])
        elif classid == 45:
            data_list.append([class_names[classid], box[2], (box[0], box[1]-2)])
        elif classid == 64:
            data_list.append([class_names[classid], box[2], (box[0], box[1]-2)])
        elif classid == 65:
            data_list.append([class_names[classid], box[2], (box[0], box[1]-2)])
        elif classid == 77:
            data_list.append([class_names[classid], box[2], (box[0], box[1]-2)])
        elif classid == 79:
            data_list.append([class_names[classid], box[2], (box[0], box[1]-2)])

    return data_list

def focal_length_finder(measured_distance, real_width, width_in_rf):
    focal_length = (width_in_rf * measured_distance) / real_width
    return focal_length

# Distance finder function
def distance_finder(focal_length, real_object_width, width_in_frame):
    distance = (real_object_width * focal_length) / width_in_frame
    return distance

@app.route('/')
def index():
    return render_template('index.html')

def generate_frames():
    cap = cv.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        data = object_detector(frame)
        for d in data:
            if d[0] == 'person':
                distance = distance_finder(focal_person, PERSON_WIDTH, d[1])
                x, y = d[2]
            elif d[0] == 'cell phone':
                distance = distance_finder(focal_mobile, MOBILE_WIDTH, d[1])
                x, y = d[2]
            cv.rectangle(frame, (x, y-3), (x+150, y+23), BLACK, -1)
            cv.putText(frame, f'Dis: {round(distance,2)} inch', (x+5, y+13), FONTS, 0.48, GREEN, 2)

        ret, buffer = cv.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Reading the reference images from directory
    ref_person = cv.imread('ReferenceImages/image13.png')
    ref_mobile = cv.imread('ReferenceImages/image1.png')
    ref_cup = cv.imread('ReferenceImages/image2.png')
    ref_toothbrush = cv.imread('ReferenceImages/image3.png')
    ref_bowl = cv.imread('ReferenceImages/image5.png')
    ref_knife = cv.imread('ReferenceImages/image6.png')
    ref_fork = cv.imread('ReferenceImages/image7.png')
    ref_bottle = cv.imread('ReferenceImages/image8.png')
    ref_teddybear = cv.imread('ReferenceImages/image10.png')
    ref_mouse = cv.imread('ReferenceImages/image11.png')
    ref_remote = cv.imread('ReferenceImages/image12.png')

    mobile_data = object_detector(ref_mobile)
    mobile_width_in_rf = mobile_data[0][1]

    person_data = object_detector(ref_person)
    person_width_in_rf = person_data[0][1]

    cup_data = object_detector(ref_cup)
    cup_width_in_rf = cup_data[0][1]

    bowl_data = object_detector(ref_bowl)
    bowl_width_in_rf = bowl_data[0][1]

    knife_data = object_detector(ref_knife)
    knife_width_in_rf = knife_data[0][1]

    fork_data = object_detector(ref_fork)
    fork_width_in_rf = fork_data[0][1]

    bottle_data = object_detector(ref_bottle)
    bottle_width_in_rf = bottle_data[0][1]

    teddybear_data = object_detector(ref_teddybear)
    teddybear_width_in_rf = teddybear_data[0][1]

    remote_data = object_detector(ref_remote)
    remote_width_in_rf = remote_data[0][1]

    # Finding focal length
    focal_person = focal_length_finder(KNOWN_DISTANCE, PERSON_WIDTH, person_width_in_rf)
    focal_mobile = focal_length_finder(KNOWN_DISTANCE, MOBILE_WIDTH, mobile_width_in_rf)

    app.run(host='0.0.0.0', port=5000, debug=True)





























# import cv2 as cv 
# import numpy as np

# # Distance constants 
# KNOWN_DISTANCE = 45 #INCHES
# PERSON_WIDTH = 16 #INCHES
# MOBILE_WIDTH = 3.0 #INCHES

# # Object detector constant 
# CONFIDENCE_THRESHOLD = 0.4
# NMS_THRESHOLD = 0.3

# # colors for object detected
# COLORS = [(255,0,0),(255,0,255),(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]
# GREEN =(0,255,0)
# BLACK =(0,0,0)
# # defining fonts 
# FONTS = cv.FONT_HERSHEY_COMPLEX

# # getting class names from classes.txt file 
# class_names = []
# with open("classes.txt", "r") as f:
#     class_names = [cname.strip() for cname in f.readlines()]
# #  setttng up opencv net
# yoloNet = cv.dnn.readNet('yolov4-tiny.weights', 'yolov4-tiny.cfg')

# yoloNet.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
# yoloNet.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)

# model = cv.dnn_DetectionModel(yoloNet)
# model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)

# # object detector funciton /method
# def object_detector(image):
#     classes, scores, boxes = model.detect(image, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
#     # creating empty list to add objects data
#     data_list =[]
#     for (classid, score, box) in zip(classes, scores, boxes):
#         # define color of each, object based on its class id 
#         color= COLORS[int(classid) % len(COLORS)]
    
#         label = "%s : %f" % (class_names[classid], score)

#         # draw rectangle on and label on object
#         cv.rectangle(image, box, color, 2)
#         cv.putText(image, label, (box[0], box[1]-14), FONTS, 0.5, color, 2)
    
#         # getting the data 
#         # 1: class name  2: object width in pixels, 3: position where have to draw text(distance)
#         if classid ==0: # person class id 
#             data_list.append([class_names[classid], box[2], (box[0], box[1]-2)])
#         elif classid ==67:
#             data_list.append([class_names[classid], box[2], (box[0], box[1]-2)])
#         # if you want inclulde more classes then you have to simply add more [elif] statements here
#         # returning list containing the object data. 
#         elif classid ==39:
#             data_list.append([class_names[classid], box[2], (box[0], box[1]-2)])

#         elif classid ==41:
#             data_list.append([class_names[classid], box[2], (box[0], box[1]-2)])

#         elif classid ==42:
#             data_list.append([class_names[classid], box[2], (box[0], box[1]-2)])

#         elif classid ==43:
#             data_list.append([class_names[classid], box[2], (box[0], box[1]-2)])

#         elif classid ==45:
#             data_list.append([class_names[classid], box[2], (box[0], box[1]-2)])

#         elif classid ==64:
#             data_list.append([class_names[classid], box[2], (box[0], box[1]-2)])

#         elif classid ==65:
#             data_list.append([class_names[classid], box[2], (box[0], box[1]-2)])
        
#         elif classid ==77:
#             data_list.append([class_names[classid], box[2], (box[0], box[1]-2)])

#         elif classid ==79:
#             data_list.append([class_names[classid], box[2], (box[0], box[1]-2)])

#     return data_list

# def focal_length_finder (measured_distance, real_width, width_in_rf):
#     focal_length = (width_in_rf * measured_distance) / real_width

#     return focal_length

# # distance finder function 
# def distance_finder(focal_length, real_object_width, width_in_frmae):
#     distance = (real_object_width * focal_length) / width_in_frmae
#     return distance

# # reading the reference image from dir 
# ref_person = cv.imread('ReferenceImages/image13.png')
# ref_mobile = cv.imread('ReferenceImages/image1.png')

# ref_cup = cv.imread('ReferenceImages/image2.png')

# ref_toothbrush = cv.imread('ReferenceImages/image3.png')

# ref_bowl = cv.imread('ReferenceImages/image5.png')

# ref_knife = cv.imread('ReferenceImages/image6.png')

# ref_fork = cv.imread('ReferenceImages/image7.png')

# ref_bottle = cv.imread('ReferenceImages/image8.png')

# ref_teddybear = cv.imread('ReferenceImages/image10.png')

# ref_mouse = cv.imread('ReferenceImages/image11.png')

# ref_remote = cv.imread('ReferenceImages/image12.png')



# mobile_data = object_detector(ref_mobile)
# mobile_width_in_rf = mobile_data[0][1]

# person_data = object_detector(ref_person)
# person_width_in_rf = person_data[0][1]

# cup_data = object_detector(ref_cup)
# cup_width_in_rf = cup_data[0][1]



# bowl_data = object_detector(ref_bowl)
# bowl_width_in_rf = bowl_data[0][1]

# knife_data = object_detector(ref_knife)
# knife_width_in_rf = knife_data[0][1]

# fork_data = object_detector(ref_fork)
# fork_width_in_rf = fork_data[0][1]

# bottle_data = object_detector(ref_bottle)
# bottle_width_in_rf = bottle_data[0][1]

# teddybear_data = object_detector(ref_teddybear)
# teddybear_width_in_rf = teddybear_data[0][1]


# remote_data = object_detector(ref_remote)
# remote_width_in_rf = remote_data[0][1]


# print(f"Person width in pixels : {person_width_in_rf} mobile width in pixel: {mobile_width_in_rf} cup width in pixel: {cup_width_in_rf} bowl width in pixel: {bowl_width_in_rf} knife width in pixel: {knife_width_in_rf} fork width in pixel: {fork_width_in_rf}  bottle width in pixel: {bottle_width_in_rf} teddybear width in pixel: {teddybear_width_in_rf}  remote width in pixel: {remote_width_in_rf}")

# # finding focal length 
# focal_person = focal_length_finder(KNOWN_DISTANCE, PERSON_WIDTH, person_width_in_rf)

# focal_mobile = focal_length_finder(KNOWN_DISTANCE, MOBILE_WIDTH, mobile_width_in_rf)
# cap = cv.VideoCapture(0)
# while True:
#     ret, frame = cap.read()

#     data = object_detector(frame) 
#     for d in data:
#         if d[0] =='person':
#             distance = distance_finder(focal_person, PERSON_WIDTH, d[1])
#             x, y = d[2]
#         elif d[0] =='cell phone':
#             distance = distance_finder (focal_mobile, MOBILE_WIDTH, d[1])
#             x, y = d[2]
#         cv.rectangle(frame, (x, y-3), (x+150, y+23),BLACK,-1 )
#         cv.putText(frame, f'Dis: {round(distance,2)} inch', (x+5,y+13), FONTS, 0.48, GREEN, 2)

#     cv.imshow('frame',frame)
    
#     key = cv.waitKey(1)
#     if key ==ord('q'):
#         break
#     if cv.getWindowProperty('frame', cv.WND_PROP_VISIBLE) < 1:
#         break
# cv.destroyAllWindows()
# cap.release()

