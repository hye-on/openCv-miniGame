import cv2
import time
from itertools import count
import sys
import numpy as np
import cv2
import time
from datetime import datetime
import threading  
 
# SET THE COUNTDOWN TIMER
# for simplicity we set it to 3
# We can also take this as input
TIMER = int(5)
  
# Open the camera
cap = cv2.VideoCapture(0)
  
 
while True:

    x, y, w, h = 170, 350, 50, 50
    rc = (x, y, w, h)

    x2, y2, w2, h2 = 390, 350, 50, 50
    rc2 = (x2, y2, w2, h2) 
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
 
        while TIMER < 0:
            ret, img = cap.read()
            img = cv2.flip(img,1)
            cv2.rectangle(img, rc, (0, 0, 255), 2)
            cv2.rectangle(img, rc2, (0, 0, 255), 2)

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
            term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

            # HS 히스토그램에 대한 역투영
            frame_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            backproj = cv2.calcBackProject([frame_hsv], channels, hist, ranges, 1)
            frame_hsv2 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            backproj2 = cv2.calcBackProject([frame_hsv], channels, hist2, ranges, 1)

           # Mean Shift
            _, rc = cv2.meanShift(backproj, rc, term_crit)
            _, rc2 = cv2.meanShift(backproj2, rc2, term_crit)

            box = cv2.boxPoints(rc).astype(np.int32)
            print(box)
 
            # Display the clicked frame for 2
            # sec.You can increase time in
            # waitKey also
            cv2.imshow('a', img)
 
            # time for which image displayed
            cv2.waitKey(100)
 

 
    # Press Esc to exit
    elif k == 27:
        break

    

# close the camera
cap.release()
  
# close all the opened windows
cv2.destroyAllWindows()    

