import numpy as np
from cv2 import cv2

enemies_path = 'sprites/enemies/'


def load_gray_image(path):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    return image

enemies = {'goomba': load_gray_image(enemies_path + 'goomba-template.png'),
               'koopa_left': load_gray_image(enemies_path + 'koopa-l-template.png'),
               'koopa_right': load_gray_image(enemies_path + 'koopa-r-template.png'),
               'shell': load_gray_image(enemies_path + 'shell-template.png'),
               'piranha_open': load_gray_image(enemies_path + 'piranha-open.png'),}
               #'piranha_closed': load_gray_image(enemies_path + 'piranha-closed.png'), }
    

def match(state, template):
    state_gray = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)

    match = cv2.matchTemplate(state_gray, template, cv2.TM_CCOEFF_NORMED)

    loc = np.where(match >= 0.85)

    # * expression unpacks lists, [::-1] gets inverting positions of y with x
    for point in zip(*loc[::-1]):
        return point

    return False


def enemy_matches(state):
    n_enemies = len(enemies)

    enemies_matches = []

    for enemy_template in enemies.values():
        point = match(state, enemy_template)
        enemies_matches.append(point)
        

    return enemies_matches