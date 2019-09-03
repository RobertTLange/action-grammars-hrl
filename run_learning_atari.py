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

env_ids = ["PongNoFrameskip-v4",
           "SeaquestNoFrameskip-v4",
           "MsPacmanNoFrameskip-v4"]

env_id = env_ids[-1]
env = make_atari(env_id)
env = wrap_deepmind(env, episode_life=True, clip_rewards=True,
                    frame_stack=True, scale=True)
env = wrap_pytorch(env)

state = env.reset()
print(state.shape)

L_RATE = 0.01
USE_CUDA = False
agents, optimizer = init_agent(CNN_DDQN, L_RATE, USE_CUDA)

epsilon = 0.1
action = agents["current"].act(state, epsilon)
print(action)
