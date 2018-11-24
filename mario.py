from __future__ import print_function
import numpy as np
from cv2 import cv2
import utils
import os
import neat
import visualize
# OpenAI Gym Imports
from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT
env = gym_super_mario_bros.make('SuperMarioBros-v2')
env = BinarySpaceToDiscreteSpaceEnv(env, SIMPLE_MOVEMENT)

STEPS = 5000

# 2-input XOR inputs and expected outputs.
xor_inputs = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
xor_outputs = [(0.0,),     (1.0,),     (1.0,),     (0.0,)]


def selection(genomes, config):
    """ Runs a game for each genome and calculate its fitness function

    Arguments:
            genomes {list of neat genomes} -- current population
            config {Config} -- config module specified by the config file
    """

    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)

        env.reset()

        # score, reward(gym already calcs), coins*weight, x_pos, status(ONE HOT ENCODING: {'small', 'tall', 'fireball'}), time
        # x_pos = 40 is the initial mario state from level 1, mario starts small and initial time is 400 secs
        INPUT = [0, 0, 0, 40, 1, 0, 0, 400]
        for step in range(STEPS):
            if done:
                state = env.reset()

            state, reward, done, info = env.step(env.action_space.sample())

            # Fixes:TypeError: Layout of the output array img is incompatible with cv::Mat (step[ndims-1] != elemsize or step[1] != elemsize*nchannels)
            state = np.ascontiguousarray(state, dtype=np.uint8)

            for point in utils.enemy_matches(state):
                cv2.rectangle(
                    state, point, (point[0] + 10, point[1] + 10), (0, 255, 255), 2)

            cv2.imshow('matching', state)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()

            output = net.activate(INPUT)
            print(len(output), np.argmax(output))

            return False

            state, reward, done, info = env.step(ACTION)

        #genome.fitness -= (output[0] - xo[0]) ** 2


def run(config_file):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    # p.add_reporter(neat.Checkpointer(5))

    # Run for up to 300 generations.
    winner = p.run(selection, 1)

    # Display the winning genome.
    #print('\nBest genome:\n{!s}'.format(winner))

    # # Show output of the most fit genome against training data.
    # print('\nOutput:')
    # winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    # for xi, xo in zip(xor_inputs, xor_outputs):
    #     output = winner_net.activate(xi)
    #     print("input {!r}, expected output {!r}, got {!r}".format(xi, xo, output))

    #node_names = {-1:'A', -2: 'B', 0:'A XOR B'}
    #visualize.draw_net(config, winner, True, filename='mario.gv' ,node_names=node_names)
    #visualize.plot_stats(stats, ylog=False, view=True)
    #visualize.plot_species(stats, view=True)

    #p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-4')
    #p.run(eval_genomes, 10)


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    config_file = os.sys.argv[1]
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, config_file)
    run(config_path)
