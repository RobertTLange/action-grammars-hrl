import argparse
import math
import time
import random
import pandas as pd
import numpy as np
from collections import deque
import gym
import copy

import torch
import torch.autograd as autograd
import torch.multiprocessing as mp

from utils.atari_wrapper import make_atari, wrap_deepmind, wrap_pytorch


def command_line_dqn_grid(parent=False):
    parser = argparse.ArgumentParser(add_help=False)
    # General logging/saving and device arguments
    parser.add_argument('-roll_upd', '--ROLLOUT_EVERY', action="store",
                        default=500, type=int,
                        help='Rollout test performance after # batch updates.')
    parser.add_argument('-n_roll', '--NUM_ROLLOUTS', action="store",
                        default=5, type=int,
                        help='# rollouts for tracking learning progrees')
    parser.add_argument('-n_runs', '--RUN_TIMES', action="store",
                        default=5, type=int,
                        help='# Times to run agent learning')
    parser.add_argument('-n_upd', '--NUM_UPDATES', action="store",
                        default=100000, type=int,
                        help='# SGD updates/iterations to train for')
    parser.add_argument('-max_steps', '--MAX_STEPS', action="store",
                        default=200, type=int,
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
                        default="mlp_agent.pt", type=str,
                        help='Path to store online agents params')
    parser.add_argument('-env_id', '--ENV_ID', action="store",
                        default="dense-v0", type=str,
                        help='Name of the environment to train on!')

    # Network architecture arguments
    parser.add_argument('-input', '--INPUT_DIM', action="store",
                        default=1200, type=int, help='Input Dimension')
    parser.add_argument('-hidden', '--HIDDEN_SIZE', action="store",
                        default=128, type=int, help='Hidden Dimension')
    parser.add_argument('-num_actions', '--NUM_ACTIONS', action="store",
                        default=4, type=int, help='Number of Actions')

    parser.add_argument('-gamma', '--GAMMA', action="store",
                        default=0.99, type=float,
                        help='Discount factor')
    parser.add_argument('-l_r', '--L_RATE', action="store", default=0.001,
                        type=float, help='Save network and learning stats after # epochs')
    parser.add_argument('-train_batch', '--TRAIN_BATCH_SIZE', action="store",
                        default=32, type=int, help='# images in training batch')

    parser.add_argument('-update_upd', '--UPDATE_EVERY', action="store",
                        default=500, type=int,
                        help='Update target network after # batch updates')

    parser.add_argument('-e_start', '--EPS_START', action="store", default=1,
                        type=float, help='Start Exploration Rate')
    parser.add_argument('-e_stop', '--EPS_STOP', action="store", default=0.01,
                        type=float, help='Start Exploration Rate')
    parser.add_argument('-e_decay', '--EPS_DECAY', action="store", default=100,
                        type=float, help='Start Exploration Rate')

    parser.add_argument('-p', '--PER', action="store_true", default=False,
                        help='Perform prioritized experience replay sampling update.')
    parser.add_argument('-b_start', '--BETA_START', action="store", default=0.4,
                        type=float, help='Initial beta to start learning with.')
    parser.add_argument('-b_steps', '--BETA_STEPS', action="store", default=2000,
                        type=int, help='Number of steps until which beta is annealed to 1.')
    parser.add_argument('-alpha', '--ALPHA', action="store", default=0.6,
                        type=float, help='Temperature is priority Boltzmann distribution.')

    parser.add_argument('-agent', '--AGENT', action="store",
                        default="MLP-DQN", type=str, help='Agent model')
    parser.add_argument('-d', '--DOUBLE', action="store_true", default=False,
                        help='Perform double Q-Learning update.')
    parser.add_argument('-capacity', '--CAPACITY', action="store",
                        default=20000, type=int, help='Storage capacity of ER buffer')
    if parent:
        return parser
    else:
        return parser.parse_args()


class ReplayBuffer(object):
    def __init__(self, capacity, record_macros=False):
        self.buffer = deque(maxlen=capacity)

    def push(self, ep_id, step, state, action,
             reward, next_state, done):
        self.buffer.append((ep_id, step, state, action, reward, next_state, done))

    def sample(self, batch_size):
        ep_id, step, state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.stack(state), action, reward, np.stack(next_state), done

    def __len__(self):
        return len(self.buffer)


def epsilon_by_episode(eps_id, epsilon_start, epsilon_final, epsilon_decay):
    eps = (epsilon_final + (epsilon_start - epsilon_final)
           * math.exp(-1. * eps_id / epsilon_decay))
    return eps


def update_target(current_model, target_model):
    # Transfer parameters from current model to target model
    target_model.load_state_dict(current_model.state_dict())


def compute_td_loss(agents, optimizer, replay_buffer,
                    TRAIN_BATCH_SIZE, GAMMA, Variable, TRAIN_DOUBLE, ENV_ID):
    obs, acts, reward, next_obs, done = replay_buffer.sample(TRAIN_BATCH_SIZE)

    if ENV_ID == "dense-v0":
        # Flatten the visual fields into vectors for MLP - not needed for CNN!
        obs = [ob.flatten() for ob in obs]
        next_obs = [next_ob.flatten() for next_ob in next_obs]

    obs = Variable(torch.FloatTensor(np.float32(obs)))
    next_obs = Variable(torch.FloatTensor(np.float32(next_obs)))
    action = Variable(torch.LongTensor(acts))
    done = Variable(torch.FloatTensor(done))

    # Select either global aggregated reward if float or agent-specific if dict
    if type(reward[0]) == np.float64 or type(reward[0]) == int:
        reward = Variable(torch.FloatTensor(reward))

    q_values = agents["current"](obs)
    q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)

    if TRAIN_DOUBLE:
        next_q_values = agents["current"](next_obs)
        next_q_state_values = agents["target"](next_obs)
        next_q_value = next_q_state_values.gather(1, torch.max(next_q_values, 1)[1].unsqueeze(1)).squeeze(1)
    else:
        next_q_values = agents["target"](next_obs)
        next_q_value = next_q_values.max(1)[0]

    expected_q_value = reward + GAMMA* next_q_value * (1 - done)

    loss = (q_value - expected_q_value.detach()).pow(2).mean()

    # Perform optimization step for agent
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm(agents["current"].parameters(), 2)
    optimizer.step()

    return loss


def get_logging_stats(opt_counter, agents, GAMMA, NUM_ROLLOUTS, MAX_STEPS, ENV_ID):
    steps = []
    rew = []

    for i in range(NUM_ROLLOUTS):
        step_temp, reward_temp, buffer = rollout_episode(agents, GAMMA, MAX_STEPS, ENV_ID)
        steps.append(step_temp)
        rew.append(reward_temp)

    steps = np.array(steps)
    rew = np.array(rew)

    reward_stats = pd.DataFrame(columns=["opt_counter", "rew_mean", "rew_sd",
                                         "rew_median",
                                         "rew_10th_p", "rew_90th_p"])

    steps_stats = pd.DataFrame(columns=["opt_counter", "steps_mean", "steps_sd",
                                        "steps_median",
                                        "steps_10th_p", "steps_90th_p"])

    reward_stats.loc[0] = [opt_counter, rew.mean(), rew.std(), np.median(rew),
                           np.percentile(rew, 10), np.percentile(rew, 90)]

    steps_stats.loc[0] = [opt_counter, steps.mean(), steps.std(), np.median(steps),
                         np.percentile(steps, 10), np.percentile(steps, 90)]

    return reward_stats, steps_stats


def rollout_episode(agents, GAMMA, MAX_STEPS, ENV_ID):
    if ENV_ID == "dense-v0":
        env = gym.make(ENV_ID)
    else:
        # Wrap the ATARI env in DeepMind Wrapper
        env = make_atari(ENV_ID)
        env = wrap_deepmind(env, episode_life=True, clip_rewards=True,
                            frame_stack=True, scale=True)
        env = wrap_pytorch(env)
    # Rollout the policy for a single episode - greedy!
    replay_buffer = ReplayBuffer(capacity=5000)

    obs = env.reset()
    episode_rew = 0
    steps = 0

    while steps < MAX_STEPS:
        if ENV_ID == "dense-v0":
            action = agents["current"].act(obs.flatten(), epsilon=0.05)
        else:
            action = agents["current"].act(obs, epsilon=0.05)
        next_obs, reward, done, _ = env.step(action)
        steps += 1

        replay_buffer.push(0, steps, obs, action,
                           reward, next_obs, done)

        obs = next_obs

        episode_rew += GAMMA**(steps - 1) * reward
        if done:
            break
    return steps, episode_rew, replay_buffer.buffer


def run_multiple_times(args, run_fct):
    cpu_count = mp.cpu_count()
    gpu_count = torch.cuda.device_count()

    # Clone arguments into list & Distribute workload across GPUs
    args_across_workers = [copy.deepcopy(args) for r in range(args.RUN_TIMES)]
    if gpu_count > 0:
        gpu_counter = 0
        for r in range(args.RUN_TIMES):
            args_across_workers[r].seed = r
            args_across_workers[r].device_id = gpu_counter
            gpu_counter += 1
            if gpu_counter > gpu_count-1:
                gpu_counter = 0

    # Execute different runs/random seeds in parallel
    pool = mp.Pool(cpu_count-1)
    df_across_runs = pool.map(run_fct, args_across_workers)
    pool.close()

    # Post process results
    df_concat = pd.concat(df_across_runs)
    by_row_index = df_concat.groupby(df_concat.index)
    df_means, df_stds = by_row_index.mean(), by_row_index.std()

    if args.ENV_ID == "dense-v0":
        sfname = "results/GRIDWORLD/" + str(args.RUN_TIMES) + "_RUNS_" + str(args.AGENT) + "_" + args.SAVE_FNAME
    else:
        sfname = "results/ATARI/" + str(args.RUN_TIMES) + "_RUNS_" + str(args.AGENT) + "_" + args.SAVE_FNAME
    print("Saved agents to {}".format(sfname))
    df_means.to_csv(sfname)
    return df_means, df_stds
