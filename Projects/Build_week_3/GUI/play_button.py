from pet_food import Pet_food
from menu_buttons import buttons
import mediapipe as mp
import numpy as np
import helpers as hp

#call the class to create the food
bone = Pet_food((1, 100), 'bone', 0)
bone1 = Pet_food((1, 100), 'bone', 4)
carrot = Pet_food((1, 200), 'carrot', 1)
carrot1 = Pet_food((1, 200), 'carrot', 5)
concrete = Pet_food((0, 400), 'concrete', 2)
worm = Pet_food((0, 300), 'worm', 3)
worm1 = Pet_food((0, 300), 'worm', 6)

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

def play(cap, response, hc, image, index_fing_x, index_fing_y):

    Quit = buttons((image.shape[0]//5, image.shape[1]//10), 'Quit')
    Quit.draw_button(image)

    if Quit.position[1]-Quit.axesLength[0]<=index_fing_x<=Quit.position[1] + Quit.axesLength[0] \
                and Quit.position[0]-Quit.axesLength[1]<=index_fing_y<=Quit.position[0] + Quit.axesLength[1]:
                game_status = 0
    else:
        game_status = 1

    #  updates the item's position
    for item in Pet_food:
        if item.current_status >= 0:
            item.move_food()

    #  draws the item in new position
    for item in Pet_food:
        if item.current_status >= 0:
            img = item.draw_food(image)

    #item reach the window's edge it bounces back
    for item in Pet_food:
        if item.current_status >= 0:
            item.detect_edge(img)

    # checks if you eat the item
    for item in Pet_food:
        if item.current_status >= 0:
            item.hand_collision(hc)

    # checks if all the items were eaten
    if np.array([item.current_status<0 for item in Pet_food if item.item_type != 'concrete']).all():
        for food in Pet_food:
            if food.item_type != 'concrete':
                # food.__init__((0, 0), food.item_type, food.item_id, new_instance=False)
                hp.reinit(food)
    elif concrete.current_status < 0:
        hp.reinit(concrete)
        pass

    return image, game_status