import cv2
import numpy as np
import mediapipe as mp


img = np.zeros(shape=[512, 512, 3], dtype=np.uint8)
class Draw():
    def __init__(self,hight, width):
        self.width=width
        self.height=height
    def draw(self,animal_name):
        self.color=(33,67,101)
        if animal_name=='Dog':
            pass



dog_head_color=(33,67,101)
center=img.shape[0]//2,img.shape[1]//2

#cv2.ellipse(img,bottum_center,(len_bottom,int(len_bottom*0.6)),angle_bottom,0,180,dog_head_color,-1)
#cv2.ellipse(img,top_center,(len_top,int(len_top*0.35)),angle_top,0,360,dog_head_color,-1)
#cv2.ellipse(img,head_top,(int(len_top*(2/3)),int(len_top*0.6)),angle_top+5,180,360,dog_head_color,-1)
#cv2.ellipse(img,(head_top[0]+40,head_top[1]-int(len_top*(2/5))),(int(len_top*0.25),int(len_top*0.65)),angle_top,180,360,dog_head_color,-1)
#eyes
cv2.circle(img,center,12,(255,255,255),cv2.FILLED)


















cv2.imshow("White Blank", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
