from __future__ import print_function
import numpy as np
import argparse
from cv2 import cv2
import utils
import os
import neat
import visualize
from pickle import dump, load, HIGHEST_PROTOCOL
# OpenAI Gym Imports
from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT
env = gym_super_mario_bros.make('SuperMarioBros-v2')
env = BinarySpaceToDiscreteSpaceEnv(env, COMPLEX_MOVEMENT)

STEPS = 0
GENERATIONS = 0


def eval(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    env.reset()
    for step in range(STEPS):
        state, reward, done, info = env.step(env.action_space.sample())

        last_score = info['score']
        time_left = info['time']
        #INPUT = [x_pos, small, tall, fireball, goomba_dist, koopa_l_dist, koopa_r_dist, shell_dist, piranha_open]
        stimulus = [info['x_pos'], float(info['status'] == 'small'), float(
            info['status'] == 'tall'), float(info['status'] == 'fireball')]

        # Fixes:TypeError: Layout of the output array img is incompatible with cv::Mat (step[ndims-1] != elemsize or step[1] != elemsize*nchannels)
        state = np.ascontiguousarray(state, dtype=np.uint8)

        matches = utils.enemy_matches(state)
        distances = [(info['x_pos'] - point[0])
                     if point else 0 for point in matches[:4]]
        piranha_status = 1 if matches[4] else 0

        stimulus.extend(distances)
        stimulus.append(piranha_status)

        output = net.activate(stimulus)
        action = np.argmax(output)

        state, reward, done, info = env.step(action)

        env.render()


def selection(genome, config):
    """ Runs a game for each genome and calculate its fitness function

    Arguments:
            genomes {list of neat genomes} -- current population
            config {Config} -- config module specified by the config file

    This function will be run in parallel # # Show output of the most fit genome against training data.
    print('\nOutput:')
    by ParallelEvaluator.  It takes two
    arguments (a single genome and the genome class configuration data) and
    should return one float (that genome's fitness).
    Note that this function needs to be in module scope for multiprocessing.Pool
    (which is what ParallelEvaluator uses) to find it.  Because of this, make
    sure you check for __main__ before executing any code (as we do here in the
    last few lines in the file), otherwise you'll have made a fork bomb
    instead of a neuroevolution demo. :)
    """

    net = neat.nn.FeedForwardNetwork.create(genome, config)

    fitness = 0.0
    # score, reward(gym already calcs), coins*weight, x_pos, status(ONE HOT ENCODING: {'small', 'tall', 'fireball'}), time
    # x_pos = 40 is the initial mario state from level 1, mario starts small and initial time is 400 secs
    last_score = 0.0
    time_left = 0.0

    env.reset()
    for step in range(STEPS):
        state, reward, done, info = env.step(env.action_space.sample())

        last_score = info['score']
        time_left = info['time']
        #INPUT = [x_pos, small, tall, fireball, goomba_dist, koopa_l_dist, koopa_r_dist, shell_dist, piranha_open]
        stimulus = [info['x_pos'], float(info['status'] == 'small'), float(
            info['status'] == 'tall'), float(info['status'] == 'fireball')]

        # Fixes:TypeError: Layout of the output array img is incompatible with cv::Mat (step[ndims-1] != elemsize or step[1] != elemsize*nchannels)
        state = np.ascontiguousarray(state, dtype=np.uint8)

        matches = utils.enemy_matches(state)
        distances = [(info['x_pos'] - point[0])
                     if point else 0 for point in matches[:4]]
        piranha_status = 1 if matches[4] else 0

        stimulus.extend(distances)
        stimulus.append(piranha_status)

        output = net.activate(stimulus)
        action = np.argmax(output)

        state, reward, done, info = env.step(action)
        fitness += reward

    fitness += last_score*0.1 + time_left*0.1

    return fitness


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
    pe = neat.ParallelEvaluator(4, selection)
    best_genome = p.run(pe.evaluate, 2)

    # Display the winning genome.
    # print('\nBest genome:\n{!s}'.format(best_genome))

    #eval(best_genome, config)
    env.close()

    # Saving best genome
    print('Saving best genome to %s' % ('best_genome_g%s_s%s.pkl' % (GENERATIONS, STEPS)))
    dump(best_genome, open('best_genome_g%s_s%s.pkl' % (GENERATIONS, STEPS), mode='wb'), protocol=HIGHEST_PROTOCOL)

    #s = load(open('best_genome_g%s_s%s.pkl' % (GENERATIONS, STEPS), mode='rb'))
    #eval(s, config)
    #node_names = {-1:'A', -2: 'B', 0:'A XOR B'}
    #visualize.draw_net(config, best_genome, True, filename='mario.gv' ,node_names=node_names)
    #visualize.plot_stats(stats, ylog=False, view=True)
    #visualize.plot_species(stats, view=True)

    #p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-4')
    #p.run(eval_genomes, 10)


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-f', '--file', help='NEATs configuration filename', required=True)
    parser.add_argument(
        '-g', '--gens', help='Number of generations', required=True)
    parser.add_argument(
        '-s', '--steps', help='Number of steps of the mario gym env', required=True)
        
    #parser.add_argument('-d', '--dir', help='Configuration directory', required=True)
    args = parser.parse_args()

    GENERATIONS = int(args.gens)
    STEPS = int(args.steps)

    config_file = args.file
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir + 'config', config_file)
    run(config_path)
