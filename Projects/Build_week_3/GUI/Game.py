
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

# USEFUL VARIABLES
global_time = time.time()
# org = (20, 30)
fontScale = 0.7
color = (0, 255, 0)
thickness = 2
CHANGE_INTERVAL = 3

bone = Pet_food((1, 100), 'bone', 0)
carrot = Pet_food((1, 200), 'carrot', 1)
concrete = Pet_food((0, 400), 'concrete', 2)
worm = Pet_food((0, 300), 'worm', 3)

cap=cv2.VideoCapture(0)
cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('Frame', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
detector=hd.handDetector(detectionCon=0.8)
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


        bone.move_food()
        carrot.move_food()
        concrete.move_food()
        worm.move_food()

        img = bone.draw_food(image)
        img = carrot.draw_food(image)
        img = concrete.draw_food(image)
        img = worm.draw_food(image)

        bone.detect_edge(img)
        carrot.detect_edge(img)
        concrete.detect_edge(img)
        worm.detect_edge(img)



        bone.hand_collision(hc)
        carrot.hand_collision(hc)
        concrete.hand_collision(hc)
        worm.hand_collision(hc)




    cv2.imshow('Frame', frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()

# video = cv2.VideoCapture(0)

# while(video.isOpened()):
#     check, frame = video.read()
#     if frame is not None:
#         img = cv2.circle(frame, (100, 100), 20, (0, 255, 0), -1)
#         #img = frame
#         cv2.imshow('frame',img)
#         if cv2.waitKey(30) & 0xFF == ord('q'):
#             break
#     else:
#         break


# video.release()
# cv2.destroyAllWindows()
# cv2.waitKey(1)



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
model = torch.load('../RESNET/2Resnet_gestures_epoch9.pth', map_location='cpu')


FRAME_COUNTER = 50
predictions_list = []
state = 'idle'
player.status = 'wait'
################################################################################

###############################################################################################################################################


with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5, max_num_hands=1) as hands:
    while response:
        response, frame = vc.read()
        image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
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
