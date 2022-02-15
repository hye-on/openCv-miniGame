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
  
# Open the camera
cap = cv2.VideoCapture(0)
sum=0
count=0


#위치 바꾼부분
x, y, w, h =115, 220, 100, 100
rc = (x, y, w, h)

x2, y2, w2, h2 =370, 220 , 100, 100
rc2 = (x2, y2, w2, h2) 

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

while True:

    # x, y, w, h =115, 220, 50, 50
    # rc = (x, y, w, h)

    # x2, y2, w2, h2 =370, 220 , 50, 50
    # rc2 = (x2, y2, w2, h2) 
    # Read and display each frame
    ret, img = cap.read()
    img = cv2.flip(img,1)
    cv2.rectangle(img, rc, (0, 0, 255), 2)
    cv2.rectangle(img, rc2, (0, 0, 255), 2)

    test="Press 's' key to start the game."

    cv2.putText(img, test, (120,100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 1) 
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
            
            img = cv2.flip(img,1)
            test="Please recognize the hand inside the square."
            test2="The game is about to start!"

            cv2.putText(img, test, (70,100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 1)
            cv2.putText(img, test2, (165,140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,255), 1)
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


            if TIMER==1:
                roi = img[y:y+h, x:x+w]
                roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

                roi2 = img[y2:y2+h2, x2:x2+w2]
                roi_hsv2 = cv2.cvtColor(roi2, cv2.COLOR_BGR2HSV)

                # HS 히스토그램 계산
                channels = [0, 1]
                ranges = [0, 180, 0, 256]
                hist = cv2.calcHist([roi_hsv], channels, None, [90, 128], ranges)
                hist2 = cv2.calcHist([roi_hsv2], channels, None, [90, 128], ranges)

                # Mean Shift 알고리즘 종료 기준
                #term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
                # CamShift 알고리즘 종료 기준
                term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

             
                

        while TIMER < 0:
            ret, img = cap.read()
            ret2=ret
            img = cv2.flip(img,1)
            if not ret:
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
            backproj = cv2.calcBackProject([frame_hsv], channels, hist, ranges, 1)
            
           #삭제
            backproj2 = cv2.calcBackProject([frame_hsv], channels, hist2, ranges, 1)

           # Mean Shift
            # _, rc = cv2.meanShift(backproj, rc, term_crit)
            # _, rc2 = cv2.meanShift(backproj2, rc2, term_crit)

 
            # CamShift
            #rc 입력이자 출력
            ret, rc = cv2.CamShift(backproj, rc, term_crit)
            ret2, rc2 = cv2.CamShift(backproj2, rc2, term_crit)
            
            #거리 출력
            x1=ret[0][0]
            y1=ret[0][1]
            x2=ret2[0][0]
            y2=ret2[0][1]

            a=x2-x1
            b=y2-y1

            t3=math.sqrt((a*a)+(b*b))
            
            sum=sum+t3
            count=count+1

            test3 = "Distance: "+ str(t3)
            cv2.putText(img, test3, (60,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,255), 1)

            # 추적 결과 화면 출력
            cv2.rectangle(img, rc, (0, 0, 255), 2)
            cv2.rectangle(img, rc2, (0, 0, 255), 2)
            cv2.ellipse(img, ret, (0, 255, 0), 2)
            cv2.ellipse(img, ret2, (0, 255, 0), 2)
            
            # x키 누르면 손 사이 거리 평균 출력
            if cv2.waitKey(1) & 0xFF == ord('x'):
                test5="This round: "
                test6=str(sum/count)
                ret2, img1 = cap.read()
                img1 = cv2.flip(img1,1)
                cv2.putText(img1, test5, (150,100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,255), 3)
                cv2.putText(img1, test6, (5,200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,255), 3)   
                cv2.imshow('a1', img1)
           
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

