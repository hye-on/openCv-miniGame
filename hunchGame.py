import sys

import numpy as np

import cv2

import time

from datetime import datetime

#now = time.localtime()

 

file = open('time.txt', 'w')  # 시간 파일 저장

 

# 비디오 파일 열기

#cap = cv2.VideoCapture('camshift.avi')

 

# open webcam (웹캠 열기)

webcam = cv2.VideoCapture(0)

 

Known_distance = 60  # centimeter

Known_width = 14.3

 

#Colors

GREEN = (0, 255, 0)

RED = (0, 0, 255)

WHITE = (255, 255, 255)

fonts = cv2.FONT_HERSHEY_COMPLEX

 

cap = cv2.VideoCapture(0)  # 웹 카메라 열기

 

# face detector object

face_detector = cv2.CascadeClassifier("haarcascade_frontface.xml")

 

# focal length finder function

 

 

def FocalLength(measured_distance, real_width, width_in_rf_image):

    focal_length = (width_in_rf_image * measured_distance)/real_width

    return focal_length

 

# distance estimation function

 

 

def Distance_finder(Focal_Length, real_face_width, face_width_in_frame):

    distance = (real_face_width * Focal_Length)/face_width_in_frame

    return distance

 

 

global rc

def face_data(image):

    face_width = 0

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(gray_image, 1.3, 5)

 

    for (x, y, h, w) in faces:

        global rc

        cv2.rectangle(image, (x, y), (x+w, y+h), WHITE, 1)

        rc = (x, y, w, h)

        face_width = w

 

    return face_width

 

 

# reading reference image from directory

ref_image = cv2.imread("Ref_image.jpg")

 

ref_image_face_width = face_data(ref_image)

Focal_length_found = FocalLength(

    Known_distance, Known_width, ref_image_face_width)

print(Focal_length_found)

#cv2.imshow("Ref_image", ref_image)

 

if not webcam.isOpened():

    print("Could not open webcam")

    exit()

 

#  if not cap.isOpened():

#      print('Video open failed!')

#      sys.exit()

 

# 초기 사각형 영역: (x, y, w, h)

 

ret, frame = webcam.read()

 

"""

x, y, w, h = cv2.selectROI(frame)

rc = (x, y, w, h)

print(rc)

"""

if not ret:

    print('frame read failed!')

    sys.exit()

 

"""

roi = frame[y:y+h, x:x+w]

roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

 

 

# HS 히스토그램 계산

channels = [0, 1]

ranges = [0, 180, 0, 256]

hist = cv2.calcHist([roi_hsv], channels, None, [90, 128], ranges)

 

 

# Mean Shift 알고리즘 종료 기준

term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

"""

 

while webcam.isOpened():

 

    # 비디오 매 프레임 처리

    #while True:

    #ret, frame = webcam.read()

    _, frame = cap.read()

    

    face_width_in_frame = face_data(frame)

    # finding the distance by calling function Distance finder

    if face_width_in_frame != 0:

        Distance = Distance_finder(Focal_length_found, Known_width, face_width_in_frame)

    

    # Drwaing Text on the screen

        cv2.putText(frame, f"Distance = {Distance}", (50, 50), fonts, 0.6, (GREEN), 2)

 

        if Distance < 160:

            cv2.putText(frame, "back your step", (50, 80), fonts, 1.0, (RED), 3)

        elif Distance > 210:

            cv2.putText(frame, "front you step", (50,80), fonts, 1.0, (RED), 3)

            

    if rc[1] < 100:

        global test

        cv2.rectangle(frame, rc, (0, 225, 0), 2)

        now = time.localtime()  # 닿은 순간 시간 계산하기 위해서

        hour = now.tm_hour

        min = now.tm_min

        sec = now.tm_sec

        test = "Time : %d : %d : %d" % (hour, min, sec)

            #cv2.putText(frame, test, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 3, lineType=None, bottomLeftOrigin=False)

        file.write(test + '\n')  # txt 파일에 시간 기록.

        cv2.putText(frame, test, (50, 100),

                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

 

    # 추적 결과 화면 출력

    else:

        cv2.rectangle(frame, rc, (0, 0, 255), 2)

 

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):

        print(test)

        file.close()

        break

    #cv2.imshow("frame", frame)

    if not ret:

        break

 

    """

    # HS 히스토그램에 대한 역투영

    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    backproj = cv2.calcBackProject([frame_hsv], channels, hist, ranges, 1)

 

    # Mean Shift

    _, rc = cv2.meanShift(backproj, rc, term_crit)

 

    #draw = ImageDraw.Draw(frame)

    #print("가나다라마바")

    

    """

    

 

webcam.release()

cv2.destroyAllWindows()


