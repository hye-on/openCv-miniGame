import sys
import numpy as np
import cv2
import time
from itertools import count
import sys
import numpy as np

from datetime import datetime

from matplotlib import pyplot as plt

TIMER = int(5)


def camshift(x1,y1,x2,y2):
    
    ret, img = cap.read()
    rc = (x1+200, y1+120, 50 ,50)
    #(268, 134, 127, 175)
    #(100, 114, 151, 211)
    print(rc)
    roi = img[y1+120:y1+170, x1+200:x1+250]
    roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    # HS 히스토그램 계산
    channels = [0, 1]
    ranges = [0, 180, 0, 256]
    hist = cv2.calcHist([roi_hsv], channels, None, [90, 128], ranges)
   # hist2 = cv2.calcHist([roi_hsv2], channels, None, [90, 128], ranges)

    # CamShift 알고리즘 종료 기준
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
    while True:
        ret, img = cap.read()
        img = cv2.flip(img,1)
       # cv2.rectangle(img, rc, (0, 0, 255), 2)
        if not ret:
            break
        frame_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)      
        backproj = cv2.calcBackProject([frame_hsv], channels, hist, ranges, 1)
        ret, rc = cv2.CamShift(backproj, rc, term_crit)

        cv2.rectangle(img, rc, (0, 0, 255), 2)
        cv2.ellipse(img, ret, (0, 255, 0), 2)
        cv2.imshow('a', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows() 
                break
    

def drawBox(img, classId, conf, left, top, right, bottom):
    # Draw a bounding box.
    cv2.rectangle(img, (left, top), (right, bottom), colors[classId], 2)

    label = f'{classes[classId]}: {conf:.2f}'

    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    top = max(top, labelSize[1])
    cv2.rectangle(img, (left - 1, top - labelSize[1] - baseLine),
                  (left + labelSize[0], top), colors[classId], -1)
    cv2.putText(img, label, (left, top - baseLine), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 0, 0), 1, cv2.LINE_AA)


# 모델 & 설정 파일
model = 'mask_rcnn/frozen_inference_graph.pb'
config = 'mask_rcnn/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt.txt'
class_labels = 'mask_rcnn/coco_90.names'
confThreshold = 0.6
maskThreshold = 0.3

# 테스트 이미지 파일
img_files = ['cup4.jpg', 'cup2.jpg', 'cup3.jpg','cup5.jpg']

# 네트워크 생성
net = cv2.dnn.readNet(model, config)

if net.empty():
    print('Net open failed!')
    sys.exit()

# 클래스 이름 불러오기

classes = []
with open(class_labels, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

colors = np.random.uniform(0, 255, size=(len(classes), 3))

# 전체 레이어 이름 받아오기
'''
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
for name in layer_names:
    print(name)
'''

cap = cv2.VideoCapture(0)
  

# 실행


ret, img = cap.read()

count=0
while cap.isOpened():
    ret, img = cap.read()
    img = cv2.flip(img,1)
    if not ret:
        break
    
    test="Please set the table so that you can see the cups, cell phones, spoons, and chopsticks. If you're ready, press s."
    cv2.putText(img, test, (120,100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 1) 
    #객체 찾았으면 camshift
    if count>0:
        #cv2.rectangle(img, (x1,x2),(y1,y2), (0, 0, 255), 2)
        camshift(x1,x2,y1,y2)
    cv2.imshow('a', img)

    
    k = cv2.waitKey(125)
    if k == ord('s'):
        count=count+1
        prev = time.time()
        while True:
            blob = cv2.dnn.blobFromImage(img, swapRB=True)
            net.setInput(blob)
            boxes, masks = net.forward(['detection_out_final', 'detection_masks'])

            # boxes.shape=(1, 1, 100, 7)
            # masks.shape=(100, 90, 15, 15)

            h, w = img.shape[:2]
            numClasses = masks.shape[1]  # 90
            numDetections = boxes.shape[2]  # 100

            boxesToDraw = []
            for i in range(numDetections):
                box = boxes[0, 0, i]  # box.shape=(7,)
                mask = masks[i]  # mask.shape=(90, 15, 15)
                score = box[2]
                if score > confThreshold:
                    classId = int(box[1])
                    print(classId, classes[classId], score)

                    x1 = int(w * box[3])
                    y1 = int(h * box[4])
                    x2 = int(w * box[5])
                    y2 = int(h * box[6])

                    x1 = max(0, min(x1, w - 1))
                    y1 = max(0, min(y1, h - 1))
                    x2 = max(0, min(x2, w - 1))
                    y2 = max(0, min(y2, h - 1))
                    # if classId==0 :
                    #     classId=2
                    boxesToDraw.append([img, classId, score, x1, y1, x2, y2])
                    classMask = mask[classId]



                    # 객체별 15x15 마스크를 바운딩 박스 크기로 resize한 후, 불투명 컬러로 표시
                    classMask = cv2.resize(classMask, (x2 - x1 + 1, y2 - y1 + 1))
                    mask = (classMask > maskThreshold)

                    roi = img[y1:y2+1, x1:x2+1][mask]
                    img[y1:y2+1, x1:x2+1][mask] = (0.7 * colors[classId] + 0.3 * roi).astype(np.uint8)

            # 객체별 바운딩 박스 그리기 & 클래스 이름 표시 + 우리가 찾는 객체일때 roi에 저장
            for box in boxesToDraw:
                drawBox(*box)
                if classes[box[1]]=="person" or classes[box[1]]=="bottle":
                    x1,x2,y1,y2= box[3], box[4], box[5], box[6]
                    #roi = img[y1:y2, x1:x2]
                    #rcTwo=(x1, x2, , y2)
                   
                   # print(roi)
            
            t, _ = net.getPerfProfile()
            label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
            cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 255), 1, cv2.LINE_AA)
            
            cur = time.time()
            if cur-prev >= 1:
                prev = cur
                TIMER = TIMER-1
            cv2.imshow('img', img)
            if cur-prev<=0:
                break
           
            #cv2.waitKey()
    

    if cv2.waitKey(1) & k == ord('q'):
        cap.release()
        cv2.destroyAllWindows() 
        break
    

# for f in img_files:
#     img = cv2.imread(f)

#     if img is None:
#         continue

#     # 블롭 생성 & 추론
#     blob = cv2.dnn.blobFromImage(img, swapRB=True)
#     net.setInput(blob)
#     boxes, masks = net.forward(['detection_out_final', 'detection_masks'])

#     # boxes.shape=(1, 1, 100, 7)
#     # masks.shape=(100, 90, 15, 15)

#     h, w = img.shape[:2]
#     numClasses = masks.shape[1]  # 90
#     numDetections = boxes.shape[2]  # 100

#     boxesToDraw = []
#     for i in range(numDetections):
#         box = boxes[0, 0, i]  # box.shape=(7,)
#         mask = masks[i]  # mask.shape=(90, 15, 15)
#         score = box[2]
#         if score > confThreshold:
#             classId = int(box[1])
#             #print(classId, classes[classId], score)

#             x1 = int(w * box[3])
#             y1 = int(h * box[4])
#             x2 = int(w * box[5])
#             y2 = int(h * box[6])

#             x1 = max(0, min(x1, w - 1))
#             y1 = max(0, min(y1, h - 1))
#             x2 = max(0, min(x2, w - 1))
#             y2 = max(0, min(y2, h - 1))

#             boxesToDraw.append([img, classId, score, x1, y1, x2, y2])
#             classMask = mask[classId]

#             # 객체별 15x15 마스크를 바운딩 박스 크기로 resize한 후, 불투명 컬러로 표시
#             classMask = cv2.resize(classMask, (x2 - x1 + 1, y2 - y1 + 1))
#             mask = (classMask > maskThreshold)

#             roi = img[y1:y2+1, x1:x2+1][mask]
#             img[y1:y2+1, x1:x2+1][mask] = (0.7 * colors[classId] + 0.3 * roi).astype(np.uint8)

#     # 객체별 바운딩 박스 그리기 & 클래스 이름 표시
#     for box in boxesToDraw:
#         drawBox(*box)

#     t, _ = net.getPerfProfile()
#     label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
#     cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
#                 0.7, (0, 0, 255), 1, cv2.LINE_AA)

#     cv2.imshow('img', img)
#     cv2.waitKey()

cv2.destroyAllWindows()
