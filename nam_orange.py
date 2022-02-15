import cv2
import time
from itertools import count
import sys
import numpy as np
import cv2
import time
from datetime import datetime
import threading
from matplotlib import pyplot as plt
#from scipy.optimize import curve_fit

# SET THE COUNTDOWN TIMER
# for simplicity we set it to 3
# We can also take this as input
TIMER = int(5)

# 얼굴로 거리 측정.
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


def face_data(image):
    face_width = 0
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray_image, 1.3, 5)

    for (x, y, h, w) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), WHITE, 1)
        face_width = w

    return face_width


# reading reference image from directory
ref_image = cv2.imread("Ref_image.jpg")

ref_image_face_width = face_data(ref_image)
Focal_length_found = FocalLength(
    Known_distance, Known_width, ref_image_face_width)
print(Focal_length_found)

# Open the camera
cap = cv2.VideoCapture(0)


#위치 바꾼부분 : 초기 사각형 위치
x, y, w, h = 115, 220, 100, 100
rc = (x, y, w, h)

#x2, y2, w2, h2 = 370, 220, 100, 100
x2, y2, w2, h2 = 870, 220, 100, 100
rc2 = (x2, y2, w2, h2)

# ret 은 카메라가 프레임을 제대로 읽었는지 체크하는 것.
ret, img = cap.read()


# roi = img[y:y+h, x:x+w]
# roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

# roi2 = img[y2:y2+h2, x2:x2+w2]
# roi_hsv2 = cv2.cvtColor(roi2, cv2.COLOR_BGR2HSV)

# # HS 히스토그램 계산
# channels = [0, 1]
# ranges = [0, 180, 0, 256]
# hist = cv2.calcHist([roi_hsv], channels, None, [90, 128], ranges)
# hist2 = cv2.calcHist([roi_hsv2], channels, None, [90, 128], ranges)

# # Mean Shift 알고리즘 종료 기준
# #term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
# # CamShift 알고리즘 종료 기준
# term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

# plt.hist(img.ravel(), 256, [0,256]);
# plt.show()


# 계속 프레임 읽어들이는 것.
while True:

    # x, y, w, h =115, 220, 50, 50
    # rc = (x, y, w, h)

    # x2, y2, w2, h2 =370, 220 , 50, 50
    # rc2 = (x2, y2, w2, h2)
    # Read and display each frame
    ret, img = cap.read()
    img = cv2.flip(img, 1) # 그림 좌우 반전.
    
    # 사각 형 그리기.
    cv2.rectangle(img, rc, (0, 0, 255), 2) # 사각형을 그릴 이미지, 사각형 좌표, 색, 선두께 순.
    cv2.rectangle(img, rc2, (0, 0, 255), 2)

    test = "Press 's' key to start the game."

    # 화면에 s 키 누르라고 출력.
    cv2.putText(img, test, (120, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 1)
    cv2.imshow('a', img) # 창 하나 생성해서 띄우기

    # check for the key pressed
    k = cv2.waitKey(125) # q 누르면 종료

    # set the key for the countdown
    # to begin. Here we set q
    # if key pressed is q
    if k == ord('s'): # 누른 키가 s 라면
        prev = time.time() # 시작 시간 (초 카운트 위해서 필요.)

        while TIMER >= 0: # 타이머 5초가 0 보다 크거나 같을 때,
            ret, img = cap.read()

            img = cv2.flip(img, 1) # 좌우 반전
            test = "Please recognize the hand inside the square."
            test2 = "The game is about to start!"

            # 사각형 안에 손을 감지시키라는 문구.
            cv2.putText(img, test, (70, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 1)
            # 경기가 시작된다는 문구 출력.
            cv2.putText(img, test2, (165, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 1)
            
            # 처음 인식하는 사각형 2개 출력.
            cv2.rectangle(img, rc, (0, 0, 255), 2)
            cv2.rectangle(img, rc2, (0, 0, 255), 2)

            # Display countdown on each frame
            # specify the font and draw the
            # countdown using puttext
            font = cv2.FONT_HERSHEY_SIMPLEX
            
            # 타이머 숫자 출력
            cv2.putText(img, str(TIMER),
                        (270, 300), font,
                        4, (255, 255, 0),
                        2)

            cv2.imshow('a', img) # 창 하나 띄우기.
            cv2.waitKey(125) # q 키 누를 시 종료

            # current time
            cur = time.time() # 현재 시간. (초 카운트 위해 필요.)

            # Update and keep track of Countdown
            # if time elapsed is one second
            # than decrease the counter
            
            # 만약, 시작 시간과 현재 시간의 차이가 1초보다 크거나 같다면.
            if cur-prev >= 1:
                prev = cur
                TIMER = TIMER-1 # 감소

            # 시간이 1초라면 손 객체 인식해서 roi 에 저장. (사각형 내부에 있는 것. 색으로)
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

                # Mean Shift 알고리즘 종료 기준
                #term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
                # CamShift 알고리즘 종료 기준
                term_crit = (cv2.TERM_CRITERIA_EPS |
                             cv2.TERM_CRITERIA_COUNT, 10, 1)

        while TIMER < 0: # 타이머가 음수라면
            ret, img = cap.read()
            ret2 = ret
            img = cv2.flip(img, 1)
            if not ret: # 카메라 정상 작동 체크
                break
            #cv2.rectangle(img, rc, (0, 0, 255), 2)
            #cv2.rectangle(img, rc2, (0, 0, 255), 2)
            #관심영역이 계속 바뀌는데 while을 돌때마다 초기화 해서 제대로 인식 못했던것.
            # roi = img[y:y+h, x:x+w]
            # roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

            # roi2 = img[y2:y2+h2, x2:x2+w2]
            # roi_hsv2 = cv2.cvtColor(roi2, cv2.COLOR_BGR2HSV)

            # # HS 히스토그램 계산
            # channels = [0, 1]
            # ranges = [0, 180, 0, 256]
            # hist = cv2.calcHist([roi_hsv], channels, None, [90, 128], ranges)
            # hist2 = cv2.calcHist([roi_hsv2], channels, None, [90, 128], ranges)

            # # Mean Shift 알고리즘 종료 기준
            # term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

            # HS 히스토그램에 대한 역투영
            frame_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            backproj = cv2.calcBackProject(
                [frame_hsv], channels, hist, ranges, 1)
           #삭제
            backproj2 = cv2.calcBackProject(
                [frame_hsv], channels, hist2, ranges, 1)

           # Mean Shift
            # _, rc = cv2.meanShift(backproj, rc, term_crit)
            # _, rc2 = cv2.meanShift(backproj2, rc2, term_crit)

            # CamShift
            #rc 입력이자 출력
            # 지속적으로 추적.
            ret, rc = cv2.CamShift(backproj, rc, term_crit)
            ret2, rc2 = cv2.CamShift(backproj2, rc2, term_crit)
            
            (_,_),width,length=ret
            # 가까워질수록 커짐
            #print(width)
            #print(length)
            
            
            # 추적 결과 화면 출력
            cv2.rectangle(img, rc, (0, 0, 255), 2)
            cv2.rectangle(img, rc2, (0, 0, 255), 2)
            cv2.ellipse(img, ret, (0, 255, 0), 2)
            cv2.ellipse(img, ret2, (0, 255, 0), 2)
            
            # 얼굴로 거리 측정.
            #_, frame = cap.read()

            face_width_in_frame = face_data(img)
            # finding the distance by calling function Distance finder
            if face_width_in_frame != 0:
                Distance = Distance_finder(
                Focal_length_found, Known_width, face_width_in_frame)

            # Drwaing Text on the screen
            cv2.putText(img, f"Distance = {Distance}",
                    (50, 50), fonts, 0.6, (GREEN), 2)

            
            #cv2.putText(ret3)
            #cv2.putText(rc3)

            # Display the clicked frame for 2
            # sec.You can increase time in
            # waitKey also
            cv2.imshow('a', img)

            # time for which image displayed
            #cv2.waitKey(100)

            # Press Esc to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                break


# close the camera
cap.release()

# close all the opened windows
cv2.destroyAllWindows()
