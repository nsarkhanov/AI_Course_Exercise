
#NEEDE IMPORTS
#######################################################################################################################################
import torch
import torchvision
import time
import cv2

import numpy as np
import mediapipe as mp

from statistics import mode

from pet_food import Pet_food
from user import User

import helpers as hp
import prediction_for_model as pfm
import animal_figures as af
import handtracking as ht
#######################################################################################################################################



# Code to be copied in the main file
#######################################################################################################################################

# USEFUL VARIABLES
global_time = time.time()
# org = (20, 30)
fontScale = 0.7
color = (0, 255, 0)
thickness = 2
CHANGE_INTERVAL = 3


# Media_pipe requirements
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Access to webcam
vc = cv2.VideoCapture(0)
if vc.isOpened():
    response, frame = vc.read()

else:
    response = False

bone = Pet_food((1, 100), 'bone', 0)
bone1 = Pet_food((1, 100), 'bone', 4)
carrot = Pet_food((1, 200), 'carrot', 1)
carrot1 = Pet_food((1, 200), 'carrot', 5)
concrete = Pet_food((0, 400), 'concrete', 2)
worm = Pet_food((0, 300), 'worm', 3)
worm1 = Pet_food((0, 300), 'worm', 6)



player = User('test')
score = 0

# Code for prediction
################################################################################
# model = torch.load('../saved_models/best_acc_model4.pth', map_location='cpu')
model = torch.load('../RESNET/mobile_gestures3.pth', map_location='cpu')


FRAME_COUNTER = 50
predictions_list = []
state = 'idle'
player.status = 'wait'
################################################################################

###############################################################################################################################################

detector=ht.handDetector(detectionCon=0.8)
with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5, max_num_hands=1) as hands:
    detector=ht.handDetector(detectionCon=0.8)
    while response:
        response, frame = vc.read()

        frame_n=cv2.flip(frame,4)

        image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)

        img_n=detector.findHands(frame_n)
        list=detector.findPosition(img_n,draw=False)


        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # print(f'landmark {hand_landmarks}')
                list_land = hand_landmarks.landmark
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                hc = hp.hand_center(hand_landmarks, image)


            # Code to be copied in the main GUI file
            ##########################################################################################################################################

                if state == 'predict' and FRAME_COUNTER > 0:
                    try:
                        predictions_list.append(pfm.predict(hand_landmarks, image, model))
                        FRAME_COUNTER -= 1
                        # print(predictions_list)
                    except:
                        continue

                if len(predictions_list) == 20:
                    final_pred = mode(predictions_list)
                    print(final_pred)
                    predictions_list = []
                    FRAME_COUNTER = 50
                    state = 'idle'
                    player.status_changer(final_pred)
            
            ###########################################################################################################################################
        


        # Code to be copied in the play() file
        ################################################################################################################################################
        try:

            
            ###########################################################################################################################################
            if player.status == 'wait':
                starting_to_wait = True
                player.status_changer(-1)
            
            
            if starting_to_wait:
                waiting_time = time.time()
                starting_to_wait = False


            if player.status == 'human' and (time.time() - waiting_time) < CHANGE_INTERVAL:

                cd_scale = (time.time() - waiting_time - (time.time() - waiting_time)//1) * 4
                text = str(CHANGE_INTERVAL - int((time.time() - waiting_time)//1))
                (w_cd, h_cd), bl_cd = cv2.getTextSize(text, Pet_food.FONT, cd_scale, thickness)
                
                txt_scale = 1.5
                (w_txt, h_txt), bl_txt = cv2.getTextSize('Prepare a gesture', Pet_food.FONT, txt_scale, thickness)

                image = cv2.putText(image, 'Prepare a gesture', (image.shape[1]//2 - w_txt//2, image.shape[0]//3),
                        Pet_food.FONT, txt_scale, Pet_food.RED, thickness, cv2.LINE_AA)


                image = cv2.putText(image, str(CHANGE_INTERVAL - int((time.time() - waiting_time)//1)),
                                   (image.shape[1]//2 - w_cd//2, image.shape[0]//2 + h_cd//2),
                                   Pet_food.FONT, cd_scale, Pet_food.RED, thickness, cv2.LINE_AA)

            else:
                if player.status == 'human':
                    state = 'predict'
            ###########################################################################################################################################

            food_list = []

            for food in Pet_food:
                hp.food_left(food, food_list)
                 
            player.check_food(food_list)
            player.check_lifes()
            
            #  updates the item's position
            for food in Pet_food:
                if food.current_status >= 0:
                    food.move_food()

            #  draws the item in new position
            for food in Pet_food:
                if food.current_status >= 0:
                    img = food.draw_food(image)

            # if the item reach the window's edge it bounces back
            for food in Pet_food:
                if food.current_status >= 0:
                    food.detect_edge(img)

            # checks if you eat the item and updates score and lifes
            if player.get_user_status() != 'human' and player.get_user_status() != 'wait':
                for food in Pet_food:
                    if food.current_status >= 0:
                        food.hand_collision(hc)
                        player.score_changer(food)

            # checks if all the items were eaten
            if np.array([item.current_status<0 for item in Pet_food if item.item_type != 'concrete']).all():
                player.status = 'wait'
                for food in Pet_food:
                    if food.item_type != 'concrete':
                        hp.reinit(food)
            elif concrete.current_status < 0:
                hp.reinit(concrete)

            # Print out lifes left and score
            ############################################################################################################################################

            if player.life_color == Pet_food.BRIGHT_GREEN or int(time.time() - global_time) % 2 == 1:
                img = cv2.putText(img, 'lifes: ' + str(player.lifes), (img.shape[1] - 100, 20), Pet_food.FONT, fontScale, player.life_color, thickness)
            img = cv2.putText(img, 'Score: ' + str(player.score), (img.shape[1] - 100, 50), Pet_food.FONT, fontScale, Pet_food.GREEN, thickness)

            ############################################################################################################################################




            # Print user status on screen - TESTING ONLY!!!!!
            #############################################################################################################################################
            if player.status == 'dog':
                af.draw_dog(list, img)
            elif player.status == 'fish':
                af.draw_fish(list, img)
            elif player.status == 'rabbit':
                af.draw_rabbit(list, img)

            img = cv2.putText(img, player.get_user_status(), tuple(hc), Pet_food.FONT, 0.7, Pet_food.RED, 1)
            #############################################################################################################################################



            cv2.imshow('test', img)
            key = cv2.waitKey(1)


            # TESTIN ONLY!!!!!!!!!!
            ##############################################################################################################################################
            if player.lifes <= 0:
                key = 27
            elif key == ord('d') or key == ord('D'):
                player.status_changer(0)
            elif key == ord('f') or key == ord('F'):
                player.status_changer(1)
            elif key == ord('r') or key == ord('R'):
                player.status_changer(2)
            elif key == ord('h') or key == ord('H'):
                player.status_changer(-1)
            elif key == ord('w') or key == ord('W'):
                player.status_changer(5)
            ###############################################################################################################################################
            
            if key == 27:
                break
        except:
            continue

        
        #################################################################################################################################################

vc.release()
cv2.destroyAllWindows()