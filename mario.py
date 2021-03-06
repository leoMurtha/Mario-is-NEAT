from __future__ import print_function
import numpy as np
import argparse
from cv2 import cv2
import utils
import os
import neat
import glob
import visualize
import logging
from pickle import dump, load, HIGHEST_PROTOCOL
# OpenAI Gym Imports
from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym_super_mario_bros
#from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT
env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = BinarySpaceToDiscreteSpaceEnv(env, [['right'], ['right', 'A'],['A']])

#print(SIMPLE_MOVEMENT)

STEPS = 0
GENERATIONS = 0


def evaluate(genome_path, config_path):
    genome = load(open(genome_path, 'rb'))
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    net = neat.nn.FeedForwardNetwork.create(genome, config)

    env.reset()
    #s = 0
    #S = 400
    state, reward, done, info = env.step(1)
    while True:
        if done:
            env.reset()
        
        # Fixes:TypeError: Layout of the output array img is incompatible with cv::Mat (step[ndims-1] != elemsize or step[1] != elemsize*nchannels)
        state = np.ascontiguousarray(state, dtype=np.uint8)

        matches, mario_pos, pit_x = utils.template_matches(state)
        
        distances = [utils.euclidean_weight(
            mario_pos, match, 1, 0.15) if match and mario_pos else 0 for match in matches]

        stimulus = distances
        
        # Finding pit distance if there is a pit else pit_distance = 0
        pit_distance = 0
        if pit_x and mario_pos:
            pit_distance = pit_x - mario_pos[0]
        #print(pit_distance, distances)
        
        stimulus.append(pit_distance)
        
        output = net.activate(stimulus)
        action = np.argmax(output)

        state, reward, done, info = env.step(action)

        state = np.ascontiguousarray(state, dtype=np.uint8)
    
        
        state_r = (cv2.cvtColor(
            state, cv2.COLOR_BGR2RGB))
        
        cv2.imshow('evaluation', state_r)
        
        
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            return True


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
    state, reward, done, info = env.step(1)
    for step in range(STEPS):
        if done:
            env.reset()

        last_score = info['score']
        time_left = info['time']
        
        matches, mario_pos, pit_x = utils.template_matches(state)
        
        distances = [utils.euclidean_weight(
            mario_pos, match, 1, 0.15) if match and mario_pos else 0 for match in matches]

        stimulus = distances
        
        # Finding pit distance if there is a pit else pit_distance = 0
        pit_distance = 0
        if pit_x and mario_pos:
            pit_distance = pit_x - mario_pos[0]
        
        # Jumped the hole
        if pit_distance <= -30 and pit_distance >= -33:
            fitness += 1000

        stimulus.append(pit_distance)
        
        output = net.activate(stimulus)
        action = np.argmax(output)

        state, reward, done, info = env.step(action)

        if done:
            fitness -= 500
            break

        if info['flag_get']:
            fitness += 10000 + last_score*0.1 + time_left*0.1

        fitness += reward

    return fitness


def run(config_path, nproc=4):
    print('[STARTING EVOLUTION] GENERATIONS %s STEPS %s' %
          (GENERATIONS, STEPS))
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    # Create the population, which is the top-level object for a NEAT run.
    #population = neat.Population(config)
    latest_model = max(glob.iglob('neat-checkpoint*'), key=os.path.getctime)

    print(latest_model)
    population = neat.Checkpointer.restore_checkpoint(latest_model)

    # Add a stdout reporter to show progress in the terminal.
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)
    # Saves a checkpoint every 10 generations
    population.add_reporter(neat.Checkpointer(5))

    # Run for up to N = GENERATIONS generations.
    parallel_eval = neat.ParallelEvaluator(nproc, selection)

    best_genome = population.run(parallel_eval.evaluate, GENERATIONS)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(best_genome))
    # Saving best genome
    print('Saving best genome to genomes/%s' %
          ('best_genome_g%s_s%s.pkl' % (population.generation, STEPS)))
    dump(best_genome, open('genomes/best_genome_g%s_s%s.pkl' %
                           (population.generation, STEPS), mode='wb'), protocol=HIGHEST_PROTOCOL)

    """
    #node_names = {-1:'A', -2: 'B', 0:'A XOR B'}
    visualize.draw_net(config, best_genome, True, filename='mario.gv')
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)
    """
    #p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-4')
    #p.run(eval_genomes, 10)


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(help='sub-command help', dest='command')

    train = subparsers.add_parser('train', help='train help')
    train.add_argument(
        '-f', '--file', help='NEATs configuration filename', required=True)
    train.add_argument(
        '-n', '--ngens', type=int, help='Number of generations', required=True)
    train.add_argument(
        '-s', '--steps', type=int, help='Number of steps of the mario gym env', required=True)
    train.add_argument(
        '-p', '--nproc', type=int, help='Number of processors', default=4)

    evaluate_parser = subparsers.add_parser('evaluate', help='evaluate help')
    evaluate_parser.add_argument(
        '-f', '--file', help='NEATs configuration filename', required=True)
    evaluate_parser.add_argument(
        '-g', '--genome', help='Genome.pkl filename', required=True)

    args = parser.parse_args()

    # Common to both commands
    config_file = args.file
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir + 'config', config_file)

    # Command selection
    if args.command == 'train':
        GENERATIONS = args.ngens
        STEPS = args.steps
        run(config_path, args.nproc)
    elif args.command == 'evaluate':
        genome_path = os.path.join(local_dir + 'genomes', args.genome)
        evaluate(genome_path, config_path)

    env.close()
