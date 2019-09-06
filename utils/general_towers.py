import time
import gym
import gym_hanoi
import numpy as np
import pandas as pd
import argparse
from collections import deque
import copy

from agents.q_agent import Agent_Q
from agents.smdp_q_agent import SMDP_Agent_Q, Macro

def command_line_towers():
    parser = argparse.ArgumentParser()
    parser.add_argument('-roll_upd', '--ROLLOUT_EVERY', action="store",
                        default=100, type=int,
                        help='Rollout test performance after # batch updates.')
    parser.add_argument('-print', '--PRINT_EVERY', action="store",
                        default=2500, type=int,
                        help='#Episodes after which to print.')
    parser.add_argument('-n_runs', '--RUN_TIMES', action="store",
                        default=1, type=int,
                        help='# Times to run agent learning')
    parser.add_argument('-n_upd', '--NUM_UPDATES', action="store",
                        default=100, type=int,
                        help='# Epochs to train for')
    parser.add_argument('-n_roll', '--NUM_ROLLOUTS', action="store",
                        default=5, type=int,
                        help='# rollouts for tracking learning progrees')
    parser.add_argument('-v', '--VERBOSE', action="store_true", default=False,
                        help='Get training progress printed out')

    parser.add_argument('-n_disks', '--N_DISKS', action="store",
                        default=4, type=int,
                        help='# Disks Hanoi Environment')
    parser.add_argument('-l_type', '--LEARN_TYPE', action="store",
                        default="Q-Learning", type=str, help='Type of learning algo')
    parser.add_argument('-g_type', '--GRAMMAR_TYPE', action="store",
                        default="2-Sequitur", type=str, help='Type of grammar inference algo')
    parser.add_argument('-t_dist', '--TRANSFER_DISTANCE', action="store",
                        default=1, type=int,
                        help='# Disks Distance in optimal Grammar')
    parser.add_argument('-save_fname', '--SAVE_FNAME', action="store",
                        default="TOH.csv", type=str,
                        help='Path to store stats of MLP agent')
    return parser.parse_args()


def learning_params(l_type, NUM_DISKS=4, ONLINE=False):
    if l_type == "Q-Learning":
        params = {"ALPHA": 0.8,  # Learning rate
                  "GAMMA": 0.95,  # Discount factor
                  "LAMBDA": 0.1,  # TD(lambda) exponential decay factor
                  "EPSILON": 0.1}  # Exploration parameter

    elif l_type == "Imitation-SMDP-Q-Learning":
        params = {"ALPHA": 0.8,  # Learning rate
                  "GAMMA": 0.95,  # Discount factor
                  "LAMBDA": 0.,  # TD(lambda) exponential decay factor
                  "EPSILON": 0.1}  # Exploration parameter

    elif l_type == "Transfer-SMDP-Q-Learning":
        params = {"ALPHA": 0.8,  # Learning rate
                  "GAMMA": 0.95,  # Discount factor
                  "LAMBDA": 0.,  # TD(lambda) exponential decay factor
                  "EPSILON": 0.1}  # Exploration parameter

    elif l_type == "Online-SMDP-Q-Learning":
        params = {"ALPHA": 0.8,  # Learning rate
                  "GAMMA": 0.95,  # Discount factor
                  "LAMBDA": 0.,  # TD(lambda) exponential decay factor
                  "EPSILON": 0.1 # Exploration parameter
                  }

    if NUM_DISKS == 4:
        params["NUM_UPDATES"] = 11000
        params["MAX_STEPS"] = 500
    elif NUM_DISKS == 5:
        params["NUM_UPDATES"] = 300000
        params["MAX_STEPS"] = 2000
    elif NUM_DISKS == 6:
        params["NUM_UPDATES"] = 7000000
        params["MAX_STEPS"] = 5000

    if ONLINE:
        if NUM_DISKS == 5:
            schedule_its = np.linspace(5000, 170000, 5)
            schedule_k = [5, 4, 3, 2, 2]
        elif NUM_DISKS == 6:
            schedule_its = np.linspace(100000, 700000, 10)
            schedule_k = [7, 6, 5, 4, 3, 2, 2, 2, 2, 2]
        return schedule_k, schedule_its
    return params


class DotDic(dict):
    # Helper module to load in parameters from json and easily call them
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __deepcopy__(self, memo=None):
        return DotDic(copy.deepcopy(dict(self), memo=memo))


class ReplayBuffer(object):
    def __init__(self, capacity, record_macros=False):
        self.buffer = deque(maxlen=capacity)
        self.record_macros = record_macros

    def push(self, ep_id, state, action,
             reward, next_state, done, macro=None):
        state = state
        next_state = next_state

        if self.record_macros:
            self.buffer.append((ep_id, state, action, macro,
                                reward, next_state, done))
        else:
            self.buffer.append((ep_id, state, action,
                                reward, next_state, done))

    def push_policy(self, ep_id, state, action, next_state):
        state = state
        next_state = next_state

        if self.record_macros:
            self.buffer.append((ep_id, state, action, macro, next_state))
        else:
            self.buffer.append((ep_id, state, action, next_state))

    def sample(self, batch_size):
        if not self.record_macros:
            ep_id, state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
            return ep_id, np.concatenate(state), action, reward, np.concatenate(next_state), done

    def __len__(self):
        return len(self.buffer)


def discounted_return(rewards, gamma):
    """
    Input: List of rewards and discount factor
    Output: Single scalar - cumulative discounted reward of episode
    """
    try:
        discounted = 0.0
        last_discount = 1.0
        for reward_set in rewards:
            gamma_mask = [gamma**t for t in range(len(reward_set))]
            # len(reward_set) works if rewards is a listoflists (from planner)
            discounted += np.dot(reward_set,
                                 gamma_mask) * last_discount * gamma
            last_discount = last_discount * gamma_mask[-1]
    except TypeError:
        # didn't work, so rewards is a list of floats - no recursion.
        gamma_mask = [gamma**t for t in range(len(rewards))]
        discounted = np.dot(rewards, gamma_mask)
    return discounted


# Dictionary of optimal Sequitur extracted macro-actions
def get_optimal_macros(N, cfg_type):
    seq2_macros = {4: ["abd"],
                   5: ["bafbcd", "baf", "ec", "bc"],
                   6: ["abdaef", "abdced", "abdaef", "aedce",
                      "abdce", "abd", "ae", "ce"],
                   7: ["bafbcdbafecfbafbcdbcfecd", "bafbcdbafecf",
                      "bafecdbcfecbafbcdbcfec", "bafbcdbafec",
                      "bcfecbafbcec"]}

    seq3_macros = {4: ["abd"],
                   5: ["baf", "bc"],
                   6: ['abdaef', 'abd', 'ced', 'ae', 'ce', 'ab'],
                   7: ['bafbcd', 'baf', 'ecf', 'bcf', 'ecd', 'ba', 'dbaf',
                       'dbcf', 'ec', 'bafbc', 'bc']}

    lexis_macros = {4: ['abd'],
                    5: ['bafbcdb'],
                    6: ['abd', 'efaedce', 'abdaefabdcedabd'],
                    7: ['bafbcdbafecfbafbcdbcfecdbafbcdb', 'fec',
                        'bafbcdb', 'fecfbafecdbcfec']}

    if cfg_type == "2-Sequitur":
        macros = get_macros_from_productions(seq2_macros[N])
        return macros
    elif cfg_type == "3-Sequitur":
        macros = get_macros_from_productions(seq3_macros[N])
        return macros
    elif cfg_type == "G-Lexis":
        macros = get_macros_from_productions(lexis_macros[N])
        return macros
    else:
        raise ValueError("Provide a valid Context-Free Grammar")


def get_macros_from_productions(productions):
    macros = []
    for i in range(len(productions)):
        macros.append(Macro(productions[i]))
    return macros


def get_logging_stats(opt_counter, agent, GAMMA,
                      N_DISKS, MAX_STEPS, NUM_ROLLOUTS):
    steps = []
    rew = []

    for i in range(NUM_ROLLOUTS):
        step_temp, reward_temp, buffer = rollout_episode(agent, MAX_STEPS,
                                                         N_DISKS, GAMMA)
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


action_to_letter = {0: "a", 1: "b", 2: "c",
                    3: "d", 4: "e", 5: "f"}


def rollout_episode(agent, MAX_STEPS, N_DISKS, GAMMA,
                    record_macros=False):

    env = gym.make("Hanoi-v0")
    env.set_env_parameters(N_DISKS, env_noise=0, verbose=False)
    er_buffer = ReplayBuffer(MAX_STEPS, record_macros)

    cur_state = env.reset()
    reward_temp = []
    steps = 0

    while steps < MAX_STEPS:
        action = agent.epsilon_greedy_action(cur_state, 0.05)
        if action > 5:
            next_state, reward, done, er_buffer = macro_step(action, cur_state, agent,
                                                             env, er_buffer, 0)
        else:
            next_state, reward, done, _ = env.step(action)
            er_buffer.push(0, cur_state, action, reward, next_state, done)

        if type(reward) != list:
            reward = [reward]

        reward_temp.extend(reward)
        cur_state = next_state
        steps += len(reward)
        if done:
            break

    episode_rew = discounted_return(reward_temp, GAMMA)
    return steps, episode_rew, er_buffer.buffer


def macro_step(action, state, agent, env, er_buffer, ep_id):
    macro_id = action - 6
    agent.macros[macro_id].active = True
    rewards = []

    while agent.macros[macro_id].active:
        action = agent.macros[macro_id].follow_macro(env)

        if action is not None:
            # Macro action is allowed to take place
            next_state, reward, done, _ = env.step(action)
            rewards.append(reward)
            if er_buffer is not None:
                er_buffer.push(ep_id, state, action,
                               reward, next_state, done, macro_id)
            state = next_state
            if done: break
        else:
            # Macro action is not valid - return -1 reward
            next_state = state
            rewards.append(-1)
            done = False
            _ = None
            break

    return next_state, rewards, done, er_buffer
