import argparse
import math
import random
import numpy as np
from collections import deque

import torch
import torch.autograd as autograd


def command_line_dqn():
    parser = argparse.ArgumentParser()
    parser.add_argument('-roll_eps', '--ROLLOUT_EVERY', action="store",
                        default=10, type=int,
                        help='Rollout test performance after # batch updates.')
    parser.add_argument('-save_eps', '--SAVE_EVERY', action="store",
                        default=50, type=int,
                        help='Save network and learning stats after # batch updates')
    parser.add_argument('-update_eps', '--UPDATE_EVERY', action="store",
                        default=10, type=int,
                        help='Update target network after # batch updates')
    parser.add_argument('-n_eps', '--NUM_EPISODES', action="store",
                        default=100, type=int,
                        help='# Epochs to train for')
    parser.add_argument('-n_roll', '--NUM_ROLLOUTS', action="store",
                        default=10, type=int,
                        help='# rollouts for tracking learning progrees')
    parser.add_argument('-max_steps', '--MAX_STEPS', action="store",
                        default=1000, type=int,
                        help='Max # of steps before episode terminated')
    parser.add_argument('-v', '--VERBOSE', action="store_true", default=False,
                        help='Get training progress printed out')


    parser.add_argument('-gamma', '--GAMMA', action="store",
                        default=0.9, type=float,
                        help='Discount factor')
    parser.add_argument('-l_r', '--L_RATE', action="store", default=0.001,
                        type=float, help='Save network and learning stats after # epochs')
    parser.add_argument('-e_start', '--EPS_START', action="store", default=1,
                        type=float, help='Start Exploration Rate')
    parser.add_argument('-e_stop', '--EPS_STOP', action="store", default=0.01,
                        type=float, help='Start Exploration Rate')
    parser.add_argument('-e_decay', '--EPS_DECAY', action="store", default=500,
                        type=float, help='Start Exploration Rate')



    parser.add_argument('-train_batch', '--TRAIN_BATCH_SIZE', action="store",
                        default=32, type=int, help='# images in training batch')
    parser.add_argument('-model', '--MODEL_TYPE', action="store",
                        default="architecture_1", type=str, help='FKP model')


    parser.add_argument('-device', '--device_id', action="store",
                        default=0, type=int, help='Device id on which to train')
    parser.add_argument('-agent_file', '--AGENT_FNAME', action="store",
                        default="mlp_agent.pt", type=str,
                        help='Path to store online agents params')
    parser.add_argument('-stats_file', '--STATS_FNAME', action="store",
                        default="MLP_agent_stats.txt", type=str,
                        help='Path to store stats of MLP agent')
    return parser.parse_args()


class ReplayBuffer(object):
    def __init__(self, capacity, record_macros=False):
        self.buffer = deque(maxlen=capacity)
        self.record_macros = record_macros

    def push(self, ep_id, step, state, action,
             reward, next_state, done, macro=None):
        if self.record_macros:
            self.buffer.append((ep_id, step, state, action, macro,
                                reward, next_state, done))
        else:
            self.buffer.append((ep_id, step, state, action,
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
                    TRAIN_BATCH_SIZE, GAMMA, Variable):
    obs, acts, reward, next_obs, done = replay_buffer.sample(TRAIN_BATCH_SIZE)

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
    next_q_values = agents["target"](next_obs)

    q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
    next_q_value = next_q_values.max(1)[0]
    expected_q_value = reward + GAMMA* next_q_value * (1 - done)

    loss = (q_value - expected_q_value.detach()).pow(2).mean()

    # Perform optimization step for agent
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm(agents["current"].parameters(), 0.5)
    optimizer.step()

    return loss


def get_logging_stats(env, agents, GAMMA, NUM_ROLLOUTS, MAX_STEPS):
    steps = []
    rewards = []

    for i in range(NUM_ROLLOUTS):
        step_temp, reward_temp, buffer = rollout_episode(env, agents, GAMMA, MAX_STEPS)
        steps.append(step_temp)
        rewards.append(reward_temp)

    steps = np.array(steps)
    rewards = np.array(rewards)

    reward_stats = {
        "mean": rewards.mean(),
        "sd": rewards.std(),
        "median": np.median(rewards),
        "10_percentile": np.percentile(rewards, 10),
        "90_percentile": np.percentile(rewards, 90)
    }

    steps_stats = {
        "mean": steps.mean(),
        "sd": steps.std(),
        "median": np.median(steps),
        "10_percentile": np.percentile(steps, 10),
        "90_percentile": np.percentile(steps, 90)
    }
    return reward_stats, steps_stats


def rollout_episode(env, agents, GAMMA, MAX_STEPS):
    # Rollout the policy for a single episode - greedy!
    replay_buffer = ReplayBuffer(capacity=5000)

    obs = env.reset()
    episode_rew = 0
    steps = 0

    for i in range(MAX_STEPS):
        action = agents["current"].act(obs.flatten(), epsilon=0)
        next_obs, reward, done, _ = env.step(action)

        replay_buffer.push(0, i, obs, action,
                           reward, next_obs, done)

        obs = next_obs

        episode_rew += GAMMA**i * reward
        steps += 1
        if done:
            break
    return steps, episode_rew, replay_buffer.buffer
