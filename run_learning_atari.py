import gym
import time
import numpy as np
import pandas as pd
import gridworld

import torch
import torch.autograd as autograd
import torch.multiprocessing as mp

from agents.dqn import CNN_DDQN, init_agent
from utils.general_dqn import command_line_dqn, ReplayBuffer, update_target, epsilon_by_episode
from utils.general_dqn import compute_td_loss, get_logging_stats, run_multiple_times
from utils.smdp_helpers_dqn import MacroBuffer, macro_action_exec, get_macro_from_agent
from utils.smdp_helpers_dqn import command_line_grammar_dqn

# Environment Wrapper!
from utils.atari_wrapper import make_atari, wrap_deepmind, wrap_pytorch
# Learning Algorithms
from run_learning_grid import run_dqn_learning

env_ids = ["PongNoFrameskip-v4",
           "SeaquestNoFrameskip-v4",
           "MsPacmanNoFrameskip-v4"]

if __name__ == "__main__":
    dqn_args = command_line_dqn(parent=True)
    all_args = command_line_grammar_dqn(dqn_args)

    run_dqn_learning(all_args)

    # python run_learning_atari.py --ENV_ID PongNoFrameskip-v4 --VERBOSE --RUN_TIMES 1 --AGENT CNN-Dueling-DQN
