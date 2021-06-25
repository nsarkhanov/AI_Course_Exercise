import cv2
import numpy as np
import mediapipe as mp
import math

def change_month(angle,width):
    if angle <31:
        width=width*0.3
    else:
        width=width*1.3
    return width

def eating(angle,color):
    if angle<31:
        color=(0,0,255)
    else:
        color=color
    return color

def draw_rabbit(list,img):
    color_head=(128,84,231)
    color_out=(0,0,0)
    color_inside=(255,255,255)
    x0,y0=list[0][1],list[0][2]
    x4,y4=list[4][1],list[4][2]
    x6,y6=list[6][1],list[6][2]
    x7,y7=list[7][1],list[7][2]
    x8,y8=list[8][1],list[8][2]
    x10,y10=list[10][1],list[10][2]
    x17,y17=list[17][1],list[17][2]
    slope_top= (y8 - y0) / (x8 - x0+1)
    angle=int(np.arctan(slope_top)*180/np.pi)
    center_coordinate=(x0+x6)//2,(y0+y6)//2
    len_bottom=int(math.hypot(x4 - x0,y4 - y0)*0.6)
    len_top=int(math.hypot(x8 - x0,y8 - y0)*0.65)
    uper_len=int(len_bottom*0.3)
    width_len=int(len_bottom*0.4)
    right_eye_coor=center_coordinate[0]+width_len,center_coordinate[1]-uper_len
    left_eye_coor=center_coordinate[0]-width_len,center_coordinate[1]-uper_len
    mounth_rabbit_center=center_coordinate[0],center_coordinate[1]+int(uper_len*1.3)
    eye_size=int(len_top*0.2)
    right_ear_coor=center_coordinate[0]+width_len,center_coordinate[1]-int(len_top*1.1)
    left_ear_coor=center_coordinate[0]-width_len,center_coordinate[1]-int(len_top*1.1)
    noise_center=center_coordinate[0],center_coordinate[1]+int(len_top*0.091)
    head_coor=center_coordinate[0]-int(center_coordinate[0]*0.08),center_coordinate[1]
    cv2.circle(img,head_coor,int(uper_len*3.8),color_head,cv2.FILLED)
    cv2.circle(img,right_eye_coor,eye_size,color_out,cv2.FILLED)
    cv2.circle(img,right_eye_coor,int(eye_size*0.8),color_inside,cv2.FILLED)
    cv2.circle(img,right_eye_coor,int(eye_size*0.3),color_out,cv2.FILLED)
    cv2.circle(img,left_eye_coor,eye_size,color_out,cv2.FILLED)
    cv2.circle(img,left_eye_coor,int(eye_size*0.8),color_inside,cv2.FILLED)
    cv2.circle(img,left_eye_coor,int(eye_size*0.3),color_out,cv2.FILLED)
    cv2.circle(img,noise_center,int(eye_size*0.8),color_out,cv2.FILLED)
    cv2.ellipse(img,mounth_rabbit_center,(int(change_month(angle,width_len)),int(change_month(angle,width_len)*0.2)),angle-30,0,180,color_out,-1)
    cv2.ellipse(img,mounth_rabbit_center,(int(change_month(angle,width_len))-int(change_month(angle,width_len)*0.1),int(change_month(angle,width_len)*0.1)),angle-30,0,180,eating(angle,color_inside),-1)
    cv2.ellipse(img,right_ear_coor,(int(width_len*1.9),int(width_len*0.6)),angle+65,0,360,color_head,-1)
    cv2.ellipse(img,left_ear_coor,(int(width_len*1.9),int(width_len*0.6)),angle+50,0,360,color_head,-1)
    return img

def draw_dog(list,img):
    x0,y0=list[0][1],list[0][2]
    x4,y4=list[4][1],list[4][2]
    x7,y7=list[7][1],list[7][2]
    x8,y8=list[8][1],list[8][2]
    x10,y10=list[10][1],list[10][2]
    x17,y17=list[17][1],list[17][2]
    bottom_coor=(x0+x4)//2,(y0+y4)//2-20
    upper_coor=(x0+x8)//2,(y0+y8)//2-20

    slope_bottom = (y4 - y0) / (x4 - x0+1)
    slope_top= (y8 - y0) / (x8 - x0+1)

    angle_top=int(np.arctan(slope_top)*180/np.pi)
    angle_bottom=int(np.arctan(slope_bottom)*180/np.pi)

    len_bottom=int(math.hypot(x4 - x0,y4 - y0)*0.6)
    len_top=int(math.hypot(x8 - x0,y8 - y0)*0.65)

    eye_coordinate=upper_coor[0],upper_coor[1]-30
    head_top=upper_coor[0]+30,upper_coor[1]-5

    color_head=(33,67,101)
    color_eye_out=(0,0,0)
    color_eye_inside=(255,255,255)
    cv2.ellipse(img,bottom_coor,(len_bottom,int(len_bottom*0.6)),angle_bottom,0,180,color_head,-1)
    cv2.ellipse(img,upper_coor,(len_top,int(len_top*0.35)),angle_top,0,360,color_head,-1)
    cv2.ellipse(img,head_top,(int(len_top*(2/3)),int(len_top*0.6)),angle_top+5,180,360,color_head,-1)
    cv2.ellipse(img,(head_top[0]+40,head_top[1]-int(len_top*(2/5))),(int(len_top*0.25),int(len_top*0.65)),angle_top,180,360,color_head,-1)

    cv2.circle(img,eye_coordinate,12,color_eye_out,cv2.FILLED)
    cv2.circle(img,eye_coordinate,6,color_eye_inside,cv2.FILLED)

    return img


def draw_fish(list,img):
    color=(255,255,0)
    color_eye_out=(0,0,0)
    color_eye_inside=(255,255,255)
    x0,y0=list[0][1],list[0][2]
    x4,y4=list[4][1],list[4][2]
    x7,y7=list[7][1],list[7][2]
    x8,y8=list[8][1],list[8][2]
    x10,y10=list[10][1],list[10][2]
    x17,y17=list[17][1],list[17][2]
    slope_bottom = (y4 - y0) / (x4 - x0+1)
    angle=int(np.arctan(slope_bottom)*180/np.pi)
    bottom_coor=(x0+x4)//2,(y0+y4)//2-20
    upper_coor=(x0+x8)//2,(y0+y8)//2-20
    len_bottom=int(math.hypot(x4 - x0,y4 - y0)*0.6)
    len_top=int(math.hypot(x8 - x0,y8 - y0)*0.65)
    center_coordinate=(bottom_coor[0]+upper_coor[0])//2+20,(bottom_coor[1]+upper_coor[1])//2-5
    size=len_top
    body_size=int(len_top*1.4)
    eye_size=int(len_top*0.1)

    eye_coor=center_coordinate[0]-int(len_top*0.85),center_coordinate[1]-int(len_top*0.22)
    tail_coor_upp=x0+int(x0*0.3),y0-int(y0*0.05)#center_coordinate[0]+int(body_size),center_coordinate[1]-int(body_size*0.3*0.3)
    tail_coor_down=x0+int(x0*0.2),y0+int(y0*0.05)#center_coordinate[0]+int(body_size),center_coordinate[1]+int(body_size*0.3*0.3)
    pad_coor=center_coordinate[0],center_coordinate[1]-int(body_size*0.25)

    cv2.ellipse(img,center_coordinate,(body_size,int(body_size*0.39)),angle,0,360,color,-1)
    cv2.circle(img,eye_coor,eye_size,color_eye_out,cv2.FILLED)
    cv2.circle(img,eye_coor,int(eye_size*0.8),color_eye_inside,cv2.FILLED)
    cv2.circle(img,eye_coor,int(eye_size*0.3),color_eye_out,cv2.FILLED)
    cv2.ellipse(img,pad_coor,(int(size*0.54),int(size*0.34)),angle-60,180,360,color,-1)
    cv2.ellipse(img,tail_coor_upp,(int(size*0.22),int(size*0.60)),angle+65,0,360,color,-1)
    cv2.ellipse(img,tail_coor_down,(int(size*0.22),int(size*0.60)),angle-65,0,360,color,-1)
    return img











#
# img = np.zeros([512,512,3],dtype=np.uint8)
# img.fill(255)
# center_coordinate=img.shape[0]//2,img.shape[1]//2
# head_size=100
# #cv2.ellipse(img,bottum_center,(len_bottom,int(len_bottom*0.6)),angle_bottom,0,180,dog_head_color,-1)
# #cv2.ellipse(img,top_center,(len_top,int(len_top*0.35)),angle_top,0,360,dog_head_color,-1)
# #cv2.ellipse(img,head_top,(int(len_top*(2/3)),int(len_top*0.6)),angle_top+5,180,360,dog_head_color,-1)
# #cv2.ellipse(img,(head_top[0]+40,head_top[1]-int(len_top*(2/5))),(int(len_top*0.25),int(len_top*0.65)),angle_top,180,360,dog_head_color,-1)
# #eyes
# # cv2.circle(img,center_coordinate,head_size,color_head,cv2.FILLED)
# # cv2.circle(img,center_coordinate,eye_size,color_eye,cv2.FILLED)
# # img=draw_dog(angle,slope,center_coordinates,size,img)
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# cv2.imshow("White Blank", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
