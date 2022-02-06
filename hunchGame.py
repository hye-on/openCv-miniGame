import sys
import numpy as np
import cv2
import time
from datetime import datetime
#now = time.localtime()

file = open('time.txt', 'w') # 시간 파일 저장

# 비디오 파일 열기
#cap = cv2.VideoCapture('camshift.avi')

# open webcam (웹캠 열기)
webcam = cv2.VideoCapture(0)

if not webcam.isOpened():
    print("Could not open webcam")
    exit()

#  if not cap.isOpened():
#      print('Video open failed!')
#      sys.exit()

# 초기 사각형 영역: (x, y, w, h)

ret, frame = webcam.read()

x, y, w, h = cv2.selectROI(frame)
rc = (x, y, w, h)

if not ret:
    print('frame read failed!')
    sys.exit()

roi = frame[y:y+h, x:x+w]
roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)


# HS 히스토그램 계산
channels = [0, 1]
ranges = [0, 180, 0, 256]
hist = cv2.calcHist([roi_hsv], channels, None, [90, 128], ranges)


# Mean Shift 알고리즘 종료 기준
term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)


while webcam.isOpened():

    # 비디오 매 프레임 처리
    #while True:
    ret, frame = webcam.read()

    if not ret:
        break

    # HS 히스토그램에 대한 역투영
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    backproj = cv2.calcBackProject([frame_hsv], channels, hist, ranges, 1)

    # Mean Shift
    _, rc = cv2.meanShift(backproj, rc, term_crit)

    #draw = ImageDraw.Draw(frame)
    #print("가나다라마바")
    if rc[1] == 0:
        global test
        cv2.rectangle(frame, rc, (0, 225, 0), 2)
        now = time.localtime() # 닿은 순간 시간 계산하기 위해서
        hour = now.tm_hour
        min = now.tm_min
        sec = now.tm_sec
        test = "Time : %d : %d : %d" %(hour, min, sec)
        #cv2.putText(frame, test, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 3, lineType=None, bottomLeftOrigin=False)
        file.write(test + '\n') # txt 파일에 시간 기록.
        cv2.putText(frame, test, (50,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
      
    # 추적 결과 화면 출력
    else:
        cv2.rectangle(frame, rc, (0, 0, 255), 2)
       
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print(test)
        file.close()
        break

webcam.release()
cv2.destroyAllWindows()