import time
import gym
import gym_hanoi
import numpy as np
from collections import deque

from agents.q_agent import Agent_Q
from agents.smdp_q_agent import SMDP_Agent_Q, Macro


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
def get_optimal_macros(env, N, cfg_type):
    seq_macros = {4: ["abd"],
                  5: ["bafbcd", "baf", "ec", "bc"],
                  6: ["abdaef", "abdced", "abdaef", "aedce",
                      "abdce", "abd", "ae", "ce"],
                  7: ["bafbcdbafecfbafbcdbcfecd", "bafbcdbafecf",
                      "bafecdbcfecbafbcdbcfec", "bafbcdbafec",
                      "bcfecbafbcec"]}

    lexis_macros = {4: ['abd'],
                    5: ['bafbcdb'],
                    6: ['abd', 'efaedce', 'abdaefabdcedabd'],
                    7: ['bafbcdbafecfbafbcdbcfecdbafbcdb', 'fec',
                        'bafbcdb', 'fecfbafecdbcfec']}

    if cfg_type == "Sequitur":
        macros = get_macros_from_productions(env, seq_macros[N])
        return macros
    elif cfg_type == "G-Lexis":
        macros = get_macros_from_productions(env,
                                             lexis_macros[N])
        return macros
    else:
        raise ValueError("Provide a valid Context-Free Grammar")


def get_macros_from_productions(env, productions):
    macros = []
    for i in range(len(productions)):
        macros.append(Macro(env, productions[i]))
    return macros

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
            if action > 5:
                next_state, reward, done, _ = macro_step(action, cur_state, agent,
                                                         env, None, i)
            else:
                next_state, reward, done, _ = env.step(action)
            if type(reward) != list:
                reward = [reward]

            reward_temp.extend(reward)
            cur_state = next_state
            stp_temp += len(reward)

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


action_to_letter = {0: "a", 1: "b", 2: "c",
                    3: "d", 4: "e", 5: "f"}


def get_rollout_policy(env, agent, max_steps,
                       record_macros=False, grammar=False):
    cur_state = env.reset()

    er_buffer_temp = ReplayBuffer(max_steps, record_macros)

    for s in range(max_steps):
        action = agent.greedy_action(cur_state)
        if action > 5:
            next_state, reward, done, _ = macro_step(action, cur_state, agent,
                                                     env, None, 0)
        else:
            next_state, reward, done, _ = env.step(action)

        er_buffer_temp.push_policy(s, cur_state, action, next_state)

        cur_state = next_state
        if done: break

    if grammar:
        sentence = []
        for i in range(len(er_buffer_temp.buffer)):
            action = er_buffer_temp.buffer[i][2]
            if action > 5:
                macro_id = action-6
                print(macro_id, len(agent.macros))
                for j in range(len(agent.macros[macro_id].action_seq)):
                    sentence.append(action_to_letter[agent.macros[macro_id].action_seq[j]])
            else:
                sentence.append(action_to_letter[action])
        return ''.join(sentence)
    else:
        return er_buffer_temp.buffer


def macro_step(action, state, agent, env, er_buffer, ep_id):
    macro_id = action-6
    agent.macros[macro_id].active = True
    rewards = []

    while agent.macros[macro_id].active:
        action = agent.macros[macro_id].follow_macro()
        if action is not None:
            # Macro action is allowed to take place
            next_state, reward, done, _ = env.step(action)
            rewards.append(reward)
            if er_buffer is not None:
                er_buffer.push(ep_id, state, action,
                               reward, next_state, done, macro_id)
            next_state = state
            if done: break
        else:
            # Macro action is not valid
            next_state = state
            done = False
            _ = None
            break

    return next_state, rewards, done, _
