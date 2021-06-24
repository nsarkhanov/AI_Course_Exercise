import cv2
from pet_food import Pet_food
import mediapipe as mp
import numpy as np
import helpers as hp


bone = Pet_food((1, 100), 'bone', 0)
carrot = Pet_food((1, 200), 'carrot', 1)
concrete = Pet_food((0, 400), 'concrete', 2)

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

def play(cap, response):
    with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5, max_num_hands=1) as hands: 
        #print('stage1 check')
        while response:
            response, image = cap.read()
            #print('stage2 check')
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = hands.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            list_landmarks = []
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    list_land = hand_landmarks.landmark
                    mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    hc = hp.hand_center(hand_landmarks, image)
            # img = cv2.circle(image, (100, 100), 20, (0, 255, 0), -1)

            img = cv2.circle(image, (0, 0), 20, (0, 255, 0), -1)
            img = cv2.circle(image, (0, image.shape[0]), 20, (0, 0, 255), -1)
            img = cv2.circle(image, (image.shape[1], 0), 20, (255, 0, 0), -1)
            img = cv2.circle(image, (image.shape[1], image.shape[0]), 20, (255, 255, 255), -1)


            bone.move_food()
            carrot.move_food()
            concrete.move_food()

            image = bone.draw_food(image)
            image = carrot.draw_food(image)
            image = concrete.draw_food(image)

            bone.detect_edge(image)
            carrot.detect_edge(image)
            concrete.detect_edge(image)

            bone.hand_collision(hc)
            carrot.hand_collision(hc)
            concrete.hand_collision(hc)

            #The function imencode compresses the image and stores it in the memory buffer
            # imgbytes=cv2.imencode('.png', image)[1].tobytes()   # Convert the image to PNG Bytes
            # window['-IMAGE-'].update(data=imgbytes) 

            cv2.imshow('Gaming Console', image)
            key = cv2.waitKey(1)
            if key == 27:
                cv2.destroyWindow('Gaming Console')
                break