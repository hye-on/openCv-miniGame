from configparser import LegacyInterpolation
import importlib
import cv2

cap = cv2.VideoCapture(0)

tracker = cv2.legacy.TrackerMOSSE_create()
ret,cam=cap.read()
cnt=0   #카운팅 관련 변수로 0으로 초기화
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
check=True
while(True):
    timer = cv2.getTickCount()
    ret, cam = cap.read()
    ret,bbox=tracker.update(cam)  
    print(bbox)
   
    if ret:
        drawBox(cam,bbox)
        if int(bbox[1]+bbox[3])<=460 & check:
                                 
            #300(파란색)을 기준으로 touch/ not touch를 구분
            cv2.putText(cam,"Touch!",(50,75),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
            if check:
                cnt=cnt+1  #터치를 했다고 간주하며 카운팅 값을 1증가시킨다
                check=False 
            cv2.putText(cam,str(cnt),(50,95),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
            cnt=int(cnt)  
            #success/fail
            if cnt==0:   #처음에 0으로 초기화하였으므로 이 경우는 고려하지 않도록 함
                pass
            else:    #카운팅 변수값이 0이 아닐 경우 
                if cnt%2==0:   #2로 나눈 나머지가 0일 경우, 즉 짝수일 경우
                    cv2.putText(cam,"SUCCESS!",(50,115),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2) #성공
                else:  #나머지가 그 외의 값일 경우, 즉 홀수일 경우 
                    cv2.putText(cam,"FAIL",(50,115),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,0,0),2)  #실패
            
               
        else:
            cnt=cnt
            check=True
            cv2.putText(cam,"NOT Touched!",(50,75),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2) #닿지 않았다면
            cv2.putText(cam,str(cnt),(50,95),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
            cnt=int(cnt)
            print(cnt)
            #success/fail
            if cnt==0:
                pass
            else:
                if cnt%2==0:
                    cv2.putText(cam,"SUCCESS!",(50,115),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
                else:
                    cv2.putText(cam,"FAIL",(50,115),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
            
    else:
        cv2.putText(cam,"Touch the line twice",(50,50),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
        cv2.putText(cam,"Lost",(50,75),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)

    fps = cv2.getTickFrequency()/(cv2.getTickCount()-timer)  
    #cv2.putText(cam,str(int(fps)),(50,50),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
    cv2.namedWindow("Tracking object", cv2.WINDOW_NORMAL)  #Tracking object 창 크기 조절
    cv2.line(cam,(30,460),(600,460),(255,255,255),4)
   
    cv2.imshow("Tracking object", cam)
         
        
    if cv2.waitKey(1) & 0xFF == 27: # esc 키를 누르면 닫음
        break
                     
cap.release()
cv2.destroyAllWindows()