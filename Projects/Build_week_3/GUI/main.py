import cv2
import numpy as np
import mediapipe as mp
import handtracking as hd
import time
import math
import animal_figures as af


cap=cv2.VideoCapture(0)
cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('Frame', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
detector=hd.handDetector(detectionCon=0.8)
while True:
    success,frame=cap.read()
    frame=cv2.flip(frame,4)
    img=detector.findHands(frame)
    list=detector.findPosition(img,draw=False)
    if len(list)!=0:
        #img=af.draw_dog(list,img)
        # img=af.draw_angry_dog(list,img)
        # img=af.draw_rabbit(list,img)
        img=af.draw_fish(list,img)

    cv2.imshow("Frame",frame)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()
cv2.destroyAllWindows()
