
import mediapipe as mp
import cv2
import numpy as np
from menu_buttons import buttons
import helpers as hp
from play_button import play


# Media_pipe requirements
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

game_status = 0
exit_status = 0
help_status = 0
#countdown = 0
#user_name = input(str('Please enter your name: '))

# Access to webcam
vc = cv2.VideoCapture(0)
if vc.isOpened():
    response, frame = vc.read()

else:
    response = False

#call the class by giving dimensions and name of the buttons
Start = buttons((frame.shape[0]//4, frame.shape[1]//2), 'Start')
Help = buttons((frame.shape[0]//4+80, frame.shape[1]//2), 'Help')
Exit = buttons((frame.shape[0]//4+160, frame.shape[1]//2), 'Exit')
#Quit = buttons((frame.shape[0]//5, frame.shape[1]//10), 'Quit')

with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5, max_num_hands=1) as hands:
    while response:
        response, frame = vc.read()
        #print(frame.shape)
        #helper_image = frame.copy() # we use it later to put text on image
        image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            image_height, image_width, maybe_depth = image.shape

            for hand_landmarks in results.multi_hand_landmarks:
                list_land = hand_landmarks.landmark
                index_fing_x, index_fing_y = (hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width,
                                    hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height)
                #print(index_fing_x, index_fing_y)

                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                hc = hp.hand_center(hand_landmarks, image)

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

        cv2.imshow('test', image)
        key = cv2.waitKey(1)
        if key == 27:
            break

vc.release()
cv2.destroyAllWindows()
