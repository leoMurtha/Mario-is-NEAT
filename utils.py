import numpy as np
from cv2 import cv2
import time
from math import sqrt, pow

enemies_path = 'sprites/enemies/'
misc_path = 'sprites/misc/'
mario_path = 'sprites/mario/'


def load_gray_image(path):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    return image


templates = {'goomba': load_gray_image(enemies_path + 'goomba-template.png'),
             'koopa_left': load_gray_image(enemies_path + 'koopa-l-template.png'),
             'koopa_right': load_gray_image(enemies_path + 'koopa-r-template.png'),
             'shell': load_gray_image(enemies_path + 'shell-template.png'),
             'pipe-up': load_gray_image(misc_path + 'pipe-up-f.png'), }
# 'piranha_open': load_gray_image(enemies_path + 'piranha-open.png'),}
# 'piranha_closed': load_gray_image(enemies_path + 'piranha-closed.png'), }

mario_templates = {'mario-small-left': load_gray_image(mario_path + 'mario-small-left-f.png'),
                   'mario-small-right': load_gray_image(mario_path + 'mario-small-right-f.png'),
                   'mario-big-left': load_gray_image(mario_path + 'mario-big-left-f.png'),
                   'mario-big-right': load_gray_image(mario_path + 'mario-big-right-f.png')}

pit_template = load_gray_image(misc_path + 'pit.png')#{'pit-left': load_gray_image(misc_path + 'pit-left.png'),
               # 'pit-right': load_gray_image(misc_path + 'pit-right.png')}


def roi(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, [vertices], 255)
    masked = cv2.bitwise_and(img, mask)
    return masked



def match(state, template):
    match = cv2.matchTemplate(state, template, cv2.TM_CCOEFF_NORMED)

    loc = np.where(match >= 0.8)

    # * expression unpacks lists, [::-1] gets inverting positions of y with x
    for point in zip(*loc[::-1]):
        return point

    return False

def detect_pit(state):
    state_r = state[210:240, 0:256]
    pit_detected = np.where(state_r == [167])
    
    if pit_detected[1] != []:
        return pit_detected[1][0]

    return False

def euclidean_weight(p1, p2, wx, wy):
    """
    point in R2 p1 is mario
    """
    distance = sqrt(pow((p1[0] - p2[0])*wx, 2) + pow((p1[1] - p2[1])*wy, 2))
    # Object is at marios left
    if p1[0] > p2[0]:
        return -distance
    return distance

def template_matches(state):
    state_gray = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)

    n_matches = len(templates)

    template_matches = []

    for template in templates.values():
        point = match(state_gray, template)
        template_matches.append(point)

    mario_pos = False
    for template in mario_templates.values():
        mario_pos = match(state_gray, template)
        if mario_pos:
            break

    pit_detected = detect_pit(state_gray)

    return template_matches, mario_pos, pit_detected
