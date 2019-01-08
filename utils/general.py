import numpy as np
from collections import deque


class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, ep_id, state, action, reward, next_state, done):
        state = state
        next_state = next_state

        self.buffer.append((ep_id, state, action, reward, next_state, done))

    def push_policy(self, ep_id, state, action, next_state):
        state = state
        next_state = next_state
        self.buffer.append((ep_id, state, action, next_state))

    def sample(self, batch_size):
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


def greedy_eval(env, agent, gamma, max_steps, log_episodes):
    rewards = []
    steps = []
    successes = 0

    for i in range(log_episodes):
        cur_state = env.reset()
        reward_temp = []
        stp_temp = 0

        for s in range(max_steps):
            action = agent.greedy_action(cur_state)
            next_state, reward, done, _ = env.step(action)
            reward_temp.append(reward)
            cur_state = next_state
            stp_temp += 1

            if done:
                rewards.append(discounted_return(reward_temp, gamma))
                steps.append(stp_temp)
                successes += 1
                break
    avg_steps = np.mean(steps) if len(steps) > 0 else max_steps
    sd_steps = np.std(steps) if len(steps) > 0 else 0
    avg_rewards = np.mean(rewards) if len(rewards) > 0 else 0
    sd_rewards = np.std(rewards) if len(rewards) > 0 else 0

    return avg_steps, sd_steps, avg_rewards, sd_rewards, successes/log_episodes


def get_rollout_policy(env, agent, max_steps):
    cur_state = env.reset()

    er_buffer_temp = ReplayBuffer(max_steps)

    for s in range(max_steps):
        action = agent.greedy_action(cur_state)
        next_state, reward, done, _ = env.step(action)

        er_buffer_temp.push_policy(s, cur_state, action, next_state)

        cur_state = next_state
        if done: break

    return er_buffer_temp.buffer
