import cv2
import numpy as np
import mediapipe as mp
import handtracking as hd
import time
import math

wcam,hcam= 640,480
cap=cv2.VideoCapture(0)
cap.set(3,wcam)
cap.set(4,hcam)
cTime=0
pTime=0
detector=hd.handDetector(detectionCon=0.8)
while True:
    success,frame=cap.read()
    frame=cv2.flip(frame,4)
    img=detector.findHands(frame)
    list=detector.findPosition(img,draw=False)
    if len(list)!=0:

        x0,y0=list[0][1],list[0][2]
        x4,y4=list[4][1],list[4][2]
        x8,y8=list[8][1],list[8][2]
        x17,y17=list[17][1],list[17][2]

        # cv2.circle(img,(x0,y0),8,(255,0,0),cv2.FILLED)
        # cv2.circle(img,(x4,y4),8,(255,0,0),cv2.FILLED)
        # cv2.circle(img,(x8,y8),8,(255,0,0),cv2.FILLED)
        # cv2.circle(img,(x17,y17),8,(255,0,0),cv2.FILLED)
        bottum_center=(x0+x4)//2,(y0+y4)//2
        top_center=(x0+x8)//2,(y0+y8)//2-15

        slope_bottom = (y4 - y0) / (x4 - x0)
        angle_bottom = np.arctan(slope_bottom)*180/np.pi
        angle_bottom=int(angle_bottom)
        slope_top= (y8 - y0) / (x8 - x0)
        angle_top = np.arctan(slope_top)*180/np.pi
        angle_top=int(angle_top)
        len_bottom=int(math.hypot(x4 - x0,y4 - y0)*0.6)
        len_top=int(math.hypot(x8 - x0,y8 - y0)*0.6)


        #print(angle_top,angle_bottom)
        cv2.circle(img,bottum_center,12,(255,0,0),cv2.FILLED)
        cv2.circle(img,top_center,12,(255,0,0),cv2.FILLED)
        # if bottum_center[1]-top_center[1]<50:
        #     cv2.ellipse(img,bottum_center,(len_bottom,int(len_bottom*0.4)),angle_bottom,10,190,(255,255,0),-1)
        #     cv2.ellipse(img,top_center,(len_top,int(len_top*0.35)),angle_top,170,350,(255,255,0),-1)
        #
        # else:
        cv2.ellipse(img,bottum_center,(len_bottom,int(len_bottom*0.4)),angle_bottom,0,180,(255,255,0),-1)
        cv2.ellipse(img,top_center,(len_top,int(len_top*0.25)),angle_top,0,360,(255,255,0),-1)

        # print(len_bottom)
        #cv2.circle(img,top_center,50,(255,0,0),-1)
        # cv2.line(img,(x0,y0),(x4-20,y4-20),(255,0,0),3)
        # cv2.line(img,(x0,y0),(x8-20,y8-20),(255,0,0),3)


    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime
    cv2.putText(frame,str(int(fps)),(10,50),cv2.FONT_HERSHEY_PLAIN,2,(255,0,0),3)
    cv2.imshow("Frame",frame)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()
cv2.destroyAllWindows()
