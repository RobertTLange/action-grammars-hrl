import gym
import time

import torch
import torch.autograd as autograd
import torch.multiprocessing as mp

import argparse
from utils.smdp_helpers_dqn import command_line_grammar_dqn

# Environment Wrapper!
from utils.atari_wrapper import make_atari, wrap_deepmind, wrap_pytorch
# Learning Algorithms
from run_learning_grid import run_dqn_learning, run_smdp_dqn_learning, run_online_dqn_smdp_learning
from utils.general_dqn import run_multiple_times


env_ids = ["PongNoFrameskip-v4", #6 actions
           "SeaquestNoFrameskip-v4", #18 actions
           "MsPacmanNoFrameskip-v4"]  #9 actions

def command_line_dqn_atari(parent=False):
    parser = argparse.ArgumentParser(add_help=False)
    # General logging/saving and device arguments
    parser.add_argument('-roll_upd', '--ROLLOUT_EVERY', action="store",
                        default=5000, type=int,
                        help='Rollout test performance after # batch updates.')
    parser.add_argument('-n_roll', '--NUM_ROLLOUTS', action="store",
                        default=5, type=int,
                        help='# rollouts for tracking learning progrees')
    parser.add_argument('-n_runs', '--RUN_TIMES', action="store",
                        default=5, type=int,
                        help='# Times to run agent learning')
    parser.add_argument('-n_upd', '--NUM_UPDATES', action="store",
                        default=500000, type=int,
                        help='# SGD updates/iterations to train for')
    parser.add_argument('-max_steps', '--MAX_STEPS', action="store",
                        default=2000000, type=int,
                        help='Max # of steps before episode terminated')
    parser.add_argument('-v', '--VERBOSE', action="store_true", default=False,
                        help='Get training progress printed out')
    parser.add_argument('-print', '--PRINT_EVERY', action="store",
                        default=50000, type=int,
                        help='#Episodes after which to print if verbose.')
    parser.add_argument('-s', '--SAVE', action="store_true",
                        default=False, help='Save final agents and log')
    parser.add_argument('-device', '--device_id', action="store",
                        default=0, type=int, help='Device id on which to train')
    parser.add_argument('-random_seed', '--seed', action="store",
                        default=0, type=int, help='random seed of agents')
    parser.add_argument('-fname', '--SAVE_FNAME', action="store",
                        default="temp", type=str, help='Filename to which to save logs')
    parser.add_argument('-agent_fname', '--AGENT_FNAME', action="store",
                        default="atari_ddqn_agent.pt", type=str,
                        help='Path to store online agents params')
    parser.add_argument('-env_id', '--ENV_ID', action="store",
                        default="PongNoFrameskip-v4", type=str,
                        help='Name of the environment to train on!')

    # Network architecture arguments
    parser.add_argument('-num_actions', '--NUM_ACTIONS', action="store",
                        default=6, type=int, help='Number of Actions')

    parser.add_argument('-gamma', '--GAMMA', action="store",
                        default=0.99, type=float,
                        help='Discount factor')
    parser.add_argument('-l_r', '--L_RATE', action="store", default=0.00025,
                        type=float, help='Save network and learning stats after # epochs')
    parser.add_argument('-train_batch', '--TRAIN_BATCH_SIZE', action="store",
                        default=32, type=int, help='# images in training batch')
    parser.add_argument('-update_upd', '--UPDATE_EVERY', action="store",
                        default=10000, type=int,
                        help='Update target network after # batch updates')

    parser.add_argument('-e_start', '--EPS_START', action="store", default=1,
                        type=float, help='Start Exploration Rate')
    parser.add_argument('-e_stop', '--EPS_STOP', action="store", default=0.01,
                        type=float, help='Start Exploration Rate')
    parser.add_argument('-e_decay', '--EPS_DECAY', action="store", default=1000000,
                        type=float, help='Start Exploration Rate')

    parser.add_argument('-agent', '--AGENT', action="store",
                        default="CNN-Dueling-DQN", type=str, help='Agent model')
    parser.add_argument('-d', '--DOUBLE', action="store_true", default=False,
                        help='Perform double Q-Learning update.')
    parser.add_argument('-capacity', '--CAPACITY', action="store",
                        default=1000000, type=int, help='Storage capacity of ER buffer')
    if parent:
        return parser
    else:
        return parser.parse_args()

if __name__ == "__main__":
    dqn_args = command_line_dqn_atari(parent=True)
    all_args = command_line_grammar_dqn(dqn_args)

    # python run_learning_atari.py --ENV_ID PongNoFrameskip-v4 --VERBOSE --RUN_TIMES 1 --AGENT CNN-Dueling-DQN

    if all_args.RUN_TIMES == 1:
        print("START RUNNING {} AGENT LEARNING FOR 1 TIME".format(all_args.AGENT))
        if all_args.RUN_EXPERT_GRAMMAR:
            run_smdp_dqn_learning(all_args)
        elif all_args.RUN_ONLINE_GRAMMAR:
            run_online_dqn_smdp_learning(all_args)
        else:
            run_dqn_learning(all_args)
    else:
        mp.set_start_method('forkserver', force=True)
        if all_args.RUN_EXPERT_GRAMMAR:
            run_multiple_times(all_args, run_smdp_dqn_learning)
        elif all_args.RUN_ONLINE_GRAMMAR:
            run_multiple_times(all_args, run_online_dqn_smdp_learning)
        else:
            run_multiple_times(all_args, run_dqn_learning)
