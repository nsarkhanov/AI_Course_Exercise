'''
**********-------**********

- To run this file, you need to install pysimplegui --> "pip install PySimpleGUI"
- Also install opencv as we will use the webcam.
- Create a copy of the file and play with it if you feel like, but -DONOT- modify this file.

**********-------**********
'''


import PySimpleGUI as sg
import cv2
from pet_food import Pet_food
import mediapipe as mp
import numpy as np
from play_button import play

#mediapipe requirements
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

#set the theme
sg.theme('Black')

# define the window layout
layout = [ # we can put a title in window but it messes up the video feed, so leving it for now
              [sg.Image(filename='', key='-IMAGE-')],
              [sg.Button('Play', size=(10, 1), font='Helvetica 14'),
               sg.Button('About', size=(10, 1), font='Any 14'),
               sg.Button('Exit', size=(10, 1), font='Helvetica 14'), ]]
    

# create the window
# we can decide on the title lateron.
window = sg.Window('Hand Gesture Demo Game', layout, location=(50,20)) 


cap = cv2.VideoCapture(0)

#Intial checkup for camera
if not cap.isOpened():
    print("Cannot open video")
    exit()

# players = []
# user = input("Please enter your name: ")

# # if user not in players:
# #     players.append(user)

# sg.PopupNoWait('Welcome', user,
#                 '' '' '' '',
#                 '' '' '' '',
#                 '' '' '' '',
#                 '' '' '' '',
#                 keep_on_top=True)


bone = Pet_food((1, 100), 'bone', 0)
carrot = Pet_food((1, 200), 'carrot', 1)
concrete = Pet_food((0, 400), 'concrete', 2)

while True:

    event, values = window.read(timeout=20)

    if event in ('Exit', None):
        break

    response, image = cap.read() # image is the frame

    if not response:
        print("Can't receive frame. Exiting ...")
        break        

    elif event in ('About', None):

        sg.PopupNoWait('This is a sample GUI for our Hand gesture project',
                    'To get the source code, click here: ',
                    'https://www.youtube.com/watch?v=DLzxrzFCyOs',
                    'We will try to integrate this GUI on top of our final project',
                    'Thank you for watching',
                    keep_on_top=True)  

    elif event in ('Play', None):

        play(cap, response)

            
    #print('stage3 check')
    imgbytes=cv2.imencode('.png', image)[1].tobytes()   # Convert the image to PNG Bytes
    window['-IMAGE-'].update(data=imgbytes)   # Change the Image Element to show the new image

window.close()