import cv2
import numpy as np

class buttons():

    #creates buttons
    def __init__(self, position: tuple, button_type: str):
        self.position = position
        self.button_type = button_type

    #draw buttons on frames
    def draw_button(self, image):

        if self.button_type == 'Start':
            x = self.position[1]//2
            y = self.position[0]//4

            center_coordinates = (x, y)
            axesLength = (50, 25)
            angle = 0
            startAngle = 0
            endAngle = 360
            color = (0, 0, 255)
            thickness = 2
            image = cv2.ellipse(image, center_coordinates, axesLength,
                    angle, startAngle, endAngle, color, thickness)
            

            font = cv2.FONT_HERSHEY_SIMPLEX
            bottomLeftCornerOfText = (285,127)
            fontScale              = 1
            fontColor              = (255,0,0)
            lineType               = 2

            cv2.putText(image,'Start', 
                bottomLeftCornerOfText, 
                font, 
                fontScale,
                fontColor,
                lineType)


        elif self.button_type == 'How to play':
            x = self.position[1]//2
            y = self.position[0]//4 + 80

            center_coordinates = (x, y)
            axesLength = (50, 25)
            angle = 0
            startAngle = 0
            endAngle = 360
            color = (0, 0, 255)
            thickness = 2
            image = cv2.ellipse(image, center_coordinates, axesLength,
                    angle, startAngle, endAngle, color, thickness)

        elif self.button_type == 'Exit':
            x = self.position[1]//2
            y = self.position[0]//4 + 160

            center_coordinates = (x, y)
            axesLength = (50, 25)
            angle = 0
            startAngle = 0
            endAngle = 360
            color = (0, 0, 255)
            thickness = 2
            image = cv2.ellipse(image, center_coordinates, axesLength,
                    angle, startAngle, endAngle, color, thickness)

        return image
