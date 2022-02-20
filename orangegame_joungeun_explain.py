import cv2
import time
from itertools import count
import sys
import numpy as np
import math
from datetime import datetime
import threading
from matplotlib import pyplot as plt


# SET THE COUNTDOWN TIMER
# for simplicity we set it to 3
# We can also take this as input
TIMER = int(5)
TIMER2 = int(3)

# Open the camera
cap = cv2.VideoCapture(0)
sum = 0
count = 0
i = 0
test6 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

# 얼굴로 거리 측정.
Known_distance = 60  # centimeter
Known_width = 14.3

# Haar Cascades 이용한 객체 검출 알고리즘.
face_detector = cv2.CascadeClassifier("haarcascade_frontface.xml") 

# 초점 거리 측정 함수
def FocalLength(measured_distance, real_width, width_in_rf_image):
    focal_length = (width_in_rf_image * measured_distance)/real_width
    return focal_length

# 거리 추정 함수
def Distance_finder(Focal_Length, real_face_width, face_width_in_frame):
    distance = ((real_face_width * Focal_Length)/face_width_in_frame)
    return distance

# haarcascade 로 영상에서 얼굴 객체 검출하여 w 값 리턴.
def face_data(image):
    face_width = 0
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray_image, 1.3, 5)

    for (x, y, h, w) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 255, 255), 1)
        face_width = w

    return face_width


# Known distance 를 재고 찍은 사진 읽기.
ref_image = cv2.imread("Ref_image.jpg")

# 얼굴 객체 검출
ref_image_face_width = face_data(ref_image)

# 초점 거리 측정
Focal_length_found = FocalLength(
    Known_distance, Known_width, ref_image_face_width)


#위치 바꾼부분
x, y, w, h = 115, 220, 100, 100
rc = (x, y, w, h)

x2, y2, w2, h2 = 370, 220, 100, 100
rc2 = (x2, y2, w2, h2)

ret, img = cap.read()
preRd = 0
while True:
    
    # Read and display each frame
    ret, img = cap.read()
    img = cv2.flip(img, 1)
    cv2.rectangle(img, rc, (0, 0, 255), 2)
    cv2.rectangle(img, rc2, (0, 0, 255), 2)

    test = "Press 's' key to start the game."

    cv2.putText(img, test, (120, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 1)
    cv2.imshow('a', img)

    # check for the key pressed
    k = cv2.waitKey(125)
    # set the key for the countdown
    # to begin. Here we set q
    # if key pressed is q
    if k == ord('s'):
        prev = time.time()

        while TIMER >= 0:
            ret, img = cap.read()

            img = cv2.flip(img, 1)
            test = "Please recognize the hand inside the square."
            test2 = "The game is about to start!"

            cv2.putText(img, test, (70, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 1)
            cv2.putText(img, test2, (165, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 1)
            cv2.rectangle(img, rc, (0, 0, 255), 2)
            cv2.rectangle(img, rc2, (0, 0, 255), 2)

            # Display countdown on each frame
            # specify the font and draw the
            # countdown using puttext
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, str(TIMER),
                        (270, 300), font,
                        4, (255, 255, 0),
                        2)

            cv2.imshow('a', img)
            cv2.waitKey(125)

            # current time
            cur = time.time()

            # Update and keep track of Countdown
            # if time elapsed is one second
            # than decrease the counter
            if cur-prev >= 1:
                prev = cur
                TIMER = TIMER-1

            if TIMER == 1:
                roi = img[y:y+h, x:x+w]
                roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

                roi2 = img[y2:y2+h2, x2:x2+w2]
                roi_hsv2 = cv2.cvtColor(roi2, cv2.COLOR_BGR2HSV)

                # HS 히스토그램 계산
                channels = [0, 1]
                ranges = [0, 180, 0, 256]
                hist = cv2.calcHist([roi_hsv], channels,
                                    None, [90, 128], ranges)
                hist2 = cv2.calcHist([roi_hsv2], channels,
                                     None, [90, 128], ranges)

                # CamShift 알고리즘 종료 기준
                term_crit = (cv2.TERM_CRITERIA_EPS |
                             cv2.TERM_CRITERIA_COUNT, 10, 1)

        prev2 = time.time()
        while TIMER < 0:

            ret, img = cap.read()
            ret2 = ret
            img = cv2.flip(img, 1)
            if not ret:
                break

            # HS 히스토그램에 대한 역투영
            frame_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            backproj = cv2.calcBackProject(
                [frame_hsv], channels, hist, ranges, 1)

           #삭제
            backproj2 = cv2.calcBackProject(
                [frame_hsv], channels, hist2, ranges, 1)

            # CamShift
            #rc 입력이자 출력
            ret, rc = cv2.CamShift(backproj, rc, term_crit)
            ret2, rc2 = cv2.CamShift(backproj2, rc2, term_crit)

            # 추적 결과 화면 출력
            cv2.rectangle(img, rc, (0, 0, 255), 2)
            cv2.rectangle(img, rc2, (0, 0, 255), 2)
            cv2.ellipse(img, ret, (0, 255, 0), 2)
            cv2.ellipse(img, ret2, (0, 255, 0), 2)

            # 얼굴로 거리 측정.
            face_width_in_frame = face_data(img) # 얼굴 w 값.
            hand_width = rc[3]  # 손1의 가로길이
            hand_width2 = rc2[3]  # 손2의 가로길이

            #거리 출력
            x1 = ret[0][0]  # 손1의 타원의 x좌표
            y1 = ret[0][1]  # 손1의 타원의 y좌표
            x2 = ret2[0][0]  # 손2의 타원의 x좌표
            y2 = ret2[0][1]  # x손2의 타원의 y좌표

            a = x2-x1
            b = y2-y1

            t3 = math.sqrt((a*a)+(b*b))  # 손과 손 사이의 거리

            test3 = "Distance: " + str(t3)

            # 거리 추정 함수로 거리 측정.
            Distance = Distance_finder(
                Focal_length_found, Known_width, face_width_in_frame)
            
            # 손이랑 카메라 거리
            hD = Distance_finder(Focal_length_found, Known_width, hand_width)
            hD2 = Distance_finder(Focal_length_found, Known_width, hand_width2)

            cur2 = time.time()
            if cur2-prev2 >= 1:
                prev2 = cur2
                TIMER2 = TIMER2-1
                #print(TIMER2)
                
            # 카메라에서 손까지 거리, 카메라에서 얼굴까지 거리 구하여 손사이 거리 도출
            # rd : 손 사이 거리.
            rd = math.sqrt(abs((hD*hD)-(Distance*Distance))) + \
                math.sqrt(abs((hD2*hD2)-(Distance*Distance)))

            rd2 = "Distance: " + str(rd)
            
            cv2.putText(img, rd2, (60, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 1)
            cv2.putText(img, str(TIMER2),(270, 300), font,4, (255, 255, 0),2)
            
            i2 = 0
            # x키 누르면 손 사이 거리 평균 출력 ->x
            #3초 지나면 자동 측정
            if TIMER2 < 0 & TIMER2 > -2:

                while i2 < 1:
                    test5 = "This round : "+str(i+1)

                    i2 += 2

                    if preRd > rd:
                        ret2, img1 = cap.read()
                        fail = "Fail!"
                        cv2.putText(img1, fail, (150, 300),
                                    cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 255), 5)
                        cv2.putText(img1, str(rd)+"  " + str(preRd), (150, 300),
                                    cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 255), 5)
                        #sum count초기화

                        cv2.imshow('a1', img1)
                        i += 1

                    else:
                        TIMER2 = 3
                        ret2, img1 = cap.read()
                        img1 = cv2.flip(img1, 1)
                        cv2.putText(img1, test5, (220, 100),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                        cv2.putText(img1, str(rd), (150, 200),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                        cv2.imshow('a1', img1)
                        preRd = rd
                        i += 1

            # Display the clicked frame for 2
            # sec.You can increase time in
            # waitKey also
            cv2.imshow('a', img)

            # time for which image displayed

            # Press Esc to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                break

# close the camera
    # cap.release()

    # # close all the opened windows
    # cv2.destroyAllWindows()
