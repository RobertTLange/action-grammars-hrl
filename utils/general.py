import time
import gym
import gym_hanoi
import numpy as np
from agents import Agent_Q, SMDP_Agent_Q, Macro
from utils.q_learning import q_learning
from utils.smdp_q_learning import smdp_q_learning
from utils.learning_params import learning_parameters
from collections import deque


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

    macros = []
    if cfg_type == "Sequitur":
        for i in range(len(seq_macros[N])):
            macros.append(Macro(env, seq_macros[N][i]))
        return macros
    elif cfg_type == "G-Lexis":
        for i in range(len(lexis_macros[N])):
            macros.append(Macro(env, lexis_macros[N][i]))
        return macros
    else:
        raise ValueError("Provide a valid Context-Free Grammar")


def run_learning(l_type, num_times, num_disks, num_episodes, max_steps,
                 log_episodes, log_freq, save_fname=None):
    its, steps, sd_steps, rew, sd_rew = [], [], [], [], []

    for i in range(num_times):
        tic = time.time()
        env = gym.make("Hanoi-v0")
        env.set_env_parameters(num_disks, env_noise=0, verbose=False)

        params = learning_parameters(l_type)

        if l_type == "Q-Learning":
            agent = Agent_Q(env)
            hist, er_buffer = q_learning(env, agent, num_episodes, max_steps,
                                         **params, log_freq=log_freq,
                                         log_episodes=log_episodes,
                                         verbose=False)

        elif l_type == "Imitation-SMDP-Q-Learning":
            macros = get_optimal_macros(env, num_disks, "Sequitur")
            agent = SMDP_Agent_Q(env, macros)
            hist, er_buffer = smdp_q_learning(env, agent, num_episodes, max_steps,
                                              **params,
                                              log_freq=log_freq,
                                              log_episodes=log_episodes,
                                              verbose=False)

        # Process results and append
        its_t, steps_t, sd_steps_t, rew_t, sd_rew_t =  hist[:, 0], hist[:, 1], hist[:,2], hist[:, 3], hist[:, 4]
        its.append(its_t)
        steps.append(steps_t)
        sd_steps.append(sd_steps_t)
        rew.append(rew_t)
        sd_rew.append(sd_rew_t)
        t_total = time.time() - tic
        print("{} Disks - {}: Run {}/{} Done - Time: {}".format(num_disks,
                                                            l_type,
                                                            i+1, num_times,
                                                            round(t_total, 2)))

    its = np.array(its).mean(axis=0)
    steps = np.array(steps).mean(axis=0)
    sd_steps = np.array(sd_steps).mean(axis=0)
    rew = np.array(rew).mean(axis=0)
    sd_rew = np.array(sd_rew).mean(axis=0)

    if save_fname is not None:
        out = np.array([its, steps, sd_steps, rew, sd_rew])
        np.savetxt(save_fname, out)
    return env, agent, its, steps, sd_steps, rew, sd_rew


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
        next_state, reward, done, _ = env.step(action)

        er_buffer_temp.push_policy(s, cur_state, action, next_state)

        cur_state = next_state
        if done: break

    if grammar:
        sentence = []
        for i in range(len(er_buffer_temp.buffer)):
            action = er_buffer_temp.buffer[i][2]
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
