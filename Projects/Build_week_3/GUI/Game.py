
#NEEDE IMPORTS
#######################################################################################################################################
import torch
import time
import cv2
import numpy as np
import mediapipe as mp
from statistics import mode
import handtracking as hd
import animal_figures as af
from pet_food import Pet_food
from user import User
import helpers as hp
import prediction_model as pfm


import mediapipe as mp
import cv2
import numpy as np
from menu_buttons import buttons
import helpers as hp
from play_button import play
global_time = time.time()
# org = (20, 30)
fontScale = 0.7
color = (0, 255, 0)
thickness = 2
CHANGE_INTERVAL = 3


# Media_pipe requirements
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

game_status = 0
exit_status = 0
help_status = 0
cap = cv2.VideoCapture(0)


response, frame = cap.read()


#call the class by giving dimensions and name of the buttons
Start = buttons((frame.shape[0]//4, frame.shape[1]//2), 'Start')
Help = buttons((frame.shape[0]//4+80, frame.shape[1]//2), 'Help')
Exit = buttons((frame.shape[0]//4+160, frame.shape[1]//2), 'Exit')

################################################################################
# model = torch.load('../saved_models/best_acc_model4.pth', map_location='cpu')
model = torch.load('../RESNET/2Resnet_gestures_epoch9.pth', map_location='cpu')
FRAME_COUNTER = 50
predictions_list = []
state = 'idle'
player.status = 'wait'




while True:
    success,frame=cap.read()
    frame=cv2.flip(frame,4)
    image=detector.findHands(frame)
    list=detector.findPosition(image,draw=False)
    if len(list)!=0:
        x=[ list[x][1] for x in range(len(list)) ]
        y=[ list[x][2] for x in range(len(list)) ]
        hc=int(np.average(x)+np.average(x)*0.02),int(np.average(y)+np.average(y)*0.031)
        img=af.draw_dog(list,image)
        img = cv2.circle(image, (0, 0), 20, (0, 255, 0), -1)
        img = cv2.circle(image, (0, image.shape[0]), 20, (0, 0, 255), -1)
        img = cv2.circle(image, (image.shape[1], 0), 20, (255, 0, 0), -1)
        img = cv2.circle(image, (image.shape[1], image.shape[0]), 20, (255, 255, 255), -1)

    if game_status == 0:
        #draw the buttons

        Start.draw_button(image)
        Help.draw_button(image)
        Exit.draw_button(image)

        #condition for index finger to touch play button
        if Start.position[1]-Start.axesLength[0]<=index_fing_x<=Start.position[1] + Start.axesLength[0] \
            and Start.position[0]-Start.axesLength[1]<=index_fing_y<=Start.position[0] + Start.axesLength[1]:
            game_status = 1

        #condition for index finger to touch exit and Quit button
        if Exit.position[1]-Exit.axesLength[0]<=index_fing_x<=Exit.position[1] + Exit.axesLength[0] \
            and Exit.position[0]-Exit.axesLength[1]<=index_fing_y<=Exit.position[0] + Exit.axesLength[1]:
            exit_status = 1

        #condition for index finger to touch exit button
        if Help.position[1]-Help.axesLength[0]<=index_fing_x<=Help.position[1] + Help.axesLength[0] \
            and Help.position[0]-Help.axesLength[1]<=index_fing_y<=Help.position[0] + Help.axesLength[1]:
            help_status = 1

    if game_status == 1:
        image, game_status = play(vc, response, hc, image, index_fing_x, index_fing_y) #play the game
    elif exit_status == 1:
        exit()
    elif help_status == 1:
        print('I am helping you')



    cv2.imshow('Frame', frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
