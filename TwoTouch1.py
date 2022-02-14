import cv2

cap = cv2.VideoCapture(0)

tracker = cv2.legacy.TrackerMOSSE_create()
ret,cam=cap.read()
bbox=cv2.selectROI("Tracking",cam,False)
tracker.init(cam,bbox)  #bbox생성

def drawBox(cam,bbox):
    #bbox 크기 조절 1
    x,y=int(bbox[0]),int(bbox[1])
    w,h=int(bbox[2]),int(bbox[3])
    #bbox크기 조절 2
    #w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    #h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cv2.rectangle(cam,(x,y),((x+w),(y+h)),(255,0,255),3,1)
    cv2.putText(cam,"Touch the line twice",(50,50),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)

while(True):
    timer = cv2.getTickCount()
    ret, cam = cap.read()

    ret,bbox=tracker.update(cam)  
    print(bbox)

    if ret:
        drawBox(cam,bbox)
        cnt=0
        if bbox[1]>=300:        #Problem1 bbox[1]은 y좌표는 좌측 상단이라 내가 원하는 좌우측 하단 y좌표가 필요
                                       #문제 해결시 기준 값을 460으로 수정 
            #300(파란색)을 기준으로 touch/ not touch를 구분
            cv2.putText(cam,"Touch!",(50,75),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
            cnt+=1
            cv2.putText(cam,str(cnt),(50,95),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
            int(cnt)   #Problem2  숫자가 1이상 올라가지 않는다
        else:
            cv2.putText(cam,"NOT Touched!",(50,75),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
    else:
        cv2.putText(cam,"Touch the line twice",(50,50),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
        cv2.putText(cam,"Lost",(50,75),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)

    fps = cv2.getTickFrequency()/(cv2.getTickCount()-timer)
    #cv2.putText(cam,str(int(fps)),(50,50),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
    cv2.namedWindow("Tracking object", cv2.WINDOW_NORMAL)  #Tracking object 창 크기 조절
    cv2.line(cam,(30,300),(600,300),(255,0,0),4)
    cv2.line(cam,(30,460),(600,460),(255,255,255),4)
    cv2.imshow("Tracking object", cam)
         
        
    if cv2.waitKey(1) & 0xFF == 27: # esc 키를 누르면 닫음
        break
                     
cap.release()
cv2.destroyAllWindows()