import cv2
import time
from itertools import count
import sys
import numpy as np
import cv2
import time
from datetime import datetime
import threading  

TIMER = int(5)
  


# open webcam (웹캠 열기)
webcam = cv2.VideoCapture(0)
 
if not webcam.isOpened():
    print("Could not open webcam")
    exit()

#  if not cap.isOpened():
#      print('Video open failed!')
#      sys.exit()

# 초기 사각형 영역: (x, y, w, h)
x, y, w, h = 115, 220, 100, 100
rc = (x, y, w, h)

# 초기 두번째 사각형 영역: (x2, y2, w, h)
x2, y2 =  370, 220 
rc2 = (x2, y2, w, h)

ret, frame = webcam.read()



if not ret:
    print('frame read failed!')
    sys.exit()


#1
#  back = cv2.cvtColor(back, cv2.COLOR_BGR2GRAY)
#  frame = cv2.GaussianBlur(back, (0, 0), 1.0)

# # 비디오 매 프레임 처리
# while True:
#     ret, frame = webcam.read()

#     if not ret:
#         break

#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     gray = cv2.GaussianBlur(gray, (0, 0), 1.0)

#     # 차영상 구하기 & 이진화
#     diff = cv2.absdiff(gray, back)
#     _, diff = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

#     # 레이블링을 이용하여 바운딩 박스 표시
#     cnt, _, stats, _ = cv2.connectedComponentsWithStats(diff)

#     for i in range(1, cnt):
#         x, y, w, h, s = stats[i]

#         if s < 5000:
#             continue

#         cv2.rectangle(frame, (x, y, w, h), (0, 0, 255), 2)

#     cv2.imshow('frame', frame)
#     cv2.imshow('diff', diff)

#     if cv2.waitKey(30) == 27:
#         break
##
roi = frame[y:y+h, x:x+w]
roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

roi2 = frame[y2:y2+h, x2:x2+w]
roi_hsv2 = cv2.cvtColor(roi2, cv2.COLOR_BGR2HSV)

# HS 히스토그램 계산
channels = [0, 1]
ranges = [0, 180, 0, 256]
hist = cv2.calcHist([roi_hsv], channels, None, [90, 128], ranges)

hist2 = cv2.calcHist([roi_hsv2], channels, None, [90, 128], ranges)

# Mean Shift 알고리즘 종료 기준
term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)


while webcam.isOpened():

# 비디오 매 프레임 처리
#while True:
    ret, frame = webcam.read()
    frame = cv2.flip(frame,1)
    if not ret:
        break

    # HS 히스토그램에 대한 역투영
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    backproj = cv2.calcBackProject([frame_hsv], channels, hist, ranges, 1)

    backproj2 = cv2.calcBackProject([frame_hsv], channels, hist2, ranges, 1)

    # Mean Shift
    # _, rc = cv2.meanShift(backproj, rc, term_crit)

    # _, rc2 = cv2.meanShift(backproj2, rc2, term_crit)

    # CamShift
    ret, rc = cv2.CamShift(backproj, rc, term_crit)
    ret2, rc2 = cv2.CamShift(backproj2, rc2, term_crit)
    # 추적 결과 화면 출력
    cv2.rectangle(frame, rc, (0, 0, 255), 2)
    cv2.rectangle(frame, rc2, (0, 0, 255), 2)
    cv2.ellipse(frame, ret, (0, 255, 0), 2)
    cv2.ellipse(frame, ret2, (0, 255, 0), 2)
    cv2.imshow('frame', frame)




    if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    

webcam.release()
cv2.destroyAllWindows()  