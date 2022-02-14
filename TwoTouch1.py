import cv2

cap = cv2.VideoCapture(0)

tracker = cv2.legacy.TrackerMOSSE_create()
ret,cam=cap.read()
bbox=cv2.selectROI("Tracking",cam,False)
tracker.init(cam,bbox)  #bbox생성

def drawBox(cam,bbox):
    x,y,w,h=int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3])
    cv2.rectangle(cam,(x,y),((x+w),(y+h)),(255,0,255),3,1)
    cv2.putText(cam,"Tracking",(50,75),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)

while(True):
    timer = cv2.getTickCount()
    ret, cam = cap.read()

    ret,bbox=tracker.update(cam)  
    print(bbox)

    if ret:
        drawBox(cam,bbox)
    else:
        cv2.putText(cam,"Lost",(50,75),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)



    fps = cv2.getTickFrequency()/(cv2.getTickCount()-timer)
    cv2.putText(cam,str(int(fps)),(50,50),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
    cv2.imshow("Tracking object", cam)
        
    if cv2.waitKey(1) & 0xFF == 27: # esc 키를 누르면 닫음
        break
                     
cap.release()
cv2.destroyAllWindows()