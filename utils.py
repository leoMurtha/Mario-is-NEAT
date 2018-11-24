from matching import enemy_matches
from cv2 import cv2
import numpy as np
from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT
env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = BinarySpaceToDiscreteSpaceEnv(env, SIMPLE_MOVEMENT)

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
               'piranha_open': load_gray_image(enemies_path + 'piranha-open.png'),
               'piranha_closed': load_gray_image(enemies_path + 'piranha-closed.png'), }
    

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
        if point:
            enemies_matches.append(point)

    return enemies_matches



env.reset()
done = False
for step in range(5000):

    if done:
        state = env.reset()

    state, reward, done, info = env.step(env.action_space.sample())

    # Fixes:TypeError: Layout of the output array img is incompatible with cv::Mat (step[ndims-1] != elemsize or step[1] != elemsize*nchannels)
    state = np.ascontiguousarray(state, dtype=np.uint8)

    for point in enemy_matches(state):
        cv2.rectangle(
            state, point, (point[0] + 10, point[1] + 10), (0, 255, 255), 2)

    cv2.imshow('matching', state)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()

   # env.render()

env.close()