import cv2

class buttons():

    #creates buttons
    def __init__(self, position: tuple, button_type: str):
        self.position = position
        self.button_type = button_type
        self.axesLength  = (50, 25)


    #draw buttons on frames
    def draw_button(self, image):

        x = self.position[1]
        y = self.position[0]

        axesLength = self.axesLength

        center_coordinates = (x, y)
        angle              = 0
        startAngle         = 0
        endAngle           = 360
        color              = (0, 0, 255)
        thickness          = -1

        image = cv2.ellipse(image, center_coordinates, axesLength,
                angle, startAngle, endAngle, color, thickness)
        

        font = cv2.FONT_HERSHEY_SIMPLEX
        
        fontScale              = 1
        fontColor              = (255,0,0)
        lineType               = 2
        thickness              = 2

        (label_width, label_height), baseline = cv2.getTextSize(self.button_type, font, fontScale, thickness)
        #(label_width, label_height), baseline = cv2.getTextSize(label, FONT, FONT_SCALE, FONT_THICKNESS)
        co_ord                 = (self.position[1]-label_width//2, self.position[0]+label_height//2)

        cv2.putText(image, self.button_type, 
            co_ord, 
            font, 
            fontScale,
            fontColor,
            lineType)


        return image
