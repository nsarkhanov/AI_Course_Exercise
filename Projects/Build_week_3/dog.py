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
        x7,y7=list[7][1],list[7][2]
        x8,y8=list[8][1],list[8][2]
        x10,y10=list[10][1],list[10][2]
        x17,y17=list[17][1],list[17][2]
        bottum_center=(x0+x4)//2,(y0+y4)//2-20
        top_center=(x0+x8)//2,(y0+y8)//2-20
        fish_center=(bottum_center[0]+top_center[0])//2+20,(bottum_center[1]+top_center[1])//2-5
        slope_bottom = (y4 - y0) / (x4 - x0+1)
        angle_bottom = np.arctan(slope_bottom)*180/np.pi
        angle_bottom=int(angle_bottom)
        slope_top= (y8 - y0) / (x8 - x0+1)
        angle_top = np.arctan(slope_top)*180/np.pi
        angle_top=int(angle_top)
        len_bottom=int(math.hypot(x4 - x0,y4 - y0)*0.6)
        len_top=int(math.hypot(x8 - x0,y8 - y0)*0.65)
        eye=top_center[0],top_center[1]-30
        head_top=top_center[0]+30,top_center[1]-5
        ###################################################################################################
        # creating dog
        # dog_head_color=(33,67,101)
        # cv2.ellipse(img,bottum_center,(len_bottom,int(len_bottom*0.6)),angle_bottom,0,180,dog_head_color,-1)
        # cv2.ellipse(img,top_center,(len_top,int(len_top*0.35)),angle_top,0,360,dog_head_color,-1)
        # cv2.ellipse(img,head_top,(int(len_top*(2/3)),int(len_top*0.6)),angle_top+5,180,360,dog_head_color,-1)
        # cv2.ellipse(img,(head_top[0]+40,head_top[1]-int(len_top*(2/5))),(int(len_top*0.25),int(len_top*0.65)),angle_top,180,360,dog_head_color,-1)
        # #eyes
        # cv2.circle(img,eye,12,(0,0,0),cv2.FILLED)
        # cv2.circle(img,eye,6,(255,255,255),cv2.FILLED)
        # ##############################################################################################################
        # # fish model
        # tail_center=x0+int(len_bottom*0.75),y0
        # fish_eye=x8,y8-20
        fish_color=(255,255,0)
        # cv2.ellipse(img,(fish_center[0],fish_center[1]+10),(int(len_top*1.35),int(len_top*0.6)),angle_top+5,0,360,fish_color,-1)
        # cv2.ellipse(img,tail_center,(int(len_top*0.25),int(len_top*0.65)),angle_top+60,0,360,fish_color,-1)
        # cv2.ellipse(img,tail_center,(int(len_top*0.25),int(len_top*0.65)),angle_top-60,0,360,fish_color,-1)
        # cv2.circle(img,fish_eye,12,(0,0,0),cv2.FILLED)
        # cv2.circle(img,fish_eye,6,(255,255,255),cv2.FILLED)
        # cv2.line(img,(list[4][1]-int(list[4][1]*0.1),list[4][2]-int(list[4][2]*0.1)),(list[4][1]+int(list[4][1]*0.1),list[4][2]-int(list[4][2]*0.1)),(255,255,255),4)
        ##############################################################################################################
        # rabbit model
        # cv2.circle(img,(x4,y4),12,(0,0,0),cv2.FILLED)
        cv2.circle(img,(x7,y7),12,(0,0,0),cv2.FILLED)
        cv2.circle(img,(x10,y10),12,(0,0,0),cv2.FILLED)
        # cv2.circle(img,(x4,y4),20,(0,255,0),cv2.FILLED)
        cv2.circle(img,(x4,y4+40),int(len_top*0.7),fish_color,cv2.FILLED)
        cv2.circle(img,(x4,y4-int(y4*0.2)),20,(255,255,255),cv2.FILLED)
        cv2.ellipse(img,(x7,y7),(int(len_top*0.2),int(len_top*0.7)),angle_top+95,0,360,fish_color,-1)
        cv2.ellipse(img,(x10+10,y10),(int(len_top*0.2),int(len_top*0.7)),angle_top+105,0,360,fish_color,-1)




    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime
    cv2.putText(frame,str(int(fps)),(10,50),cv2.FONT_HERSHEY_PLAIN,2,(255,0,0),3)
    cv2.imshow("Frame",frame)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()
cv2.destroyAllWindows()
