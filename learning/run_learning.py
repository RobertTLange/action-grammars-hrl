import time
import numpy as np

import gym
import gym_hanoi

from agents.q_agent import Agent_Q
from agents.smdp_q_agent import SMDP_Agent_Q, Macro

from learning.q_learning import q_learning
from learning.smdp_q_learning import smdp_q_learning, smdp_q_online_learning
from learning.learning_params import learning_parameters

from utils.general import get_optimal_macros


def run_learning(l_type, num_times, num_disks, num_episodes, max_steps,
                 log_episodes, log_freq, transfer_distance=None,
                 save_fname=None):
    its, steps, sd_steps, rew, sd_rew = [], [], [], [], []

    env = gym.make("Hanoi-v0")
    env.set_env_parameters(num_disks, env_noise=0, verbose=False)
    params = learning_parameters(l_type)

    if l_type == "Q-Learning":
        agent = Agent_Q(env)
    elif l_type == "Imitation-SMDP-Q-Learning":
        macros = get_optimal_macros(env, num_disks, "Sequitur")
        agent = SMDP_Agent_Q(env, macros)
    elif l_type == "Transfer-SMDP-Q-Learning":
        macros = get_optimal_macros(env,
                                    num_disks - transfer_distance,
                                    "Sequitur")
        agent = SMDP_Agent_Q(env, macros)

    for i in range(num_times):
        tic = time.time()
        # Reset values to 0 initialization without having to recompute mov_map
        agent.reset_values()
        if l_type == "Q-Learning":
            hist, er_buffer = q_learning(env, agent, num_episodes, max_steps,
                                         **params, log_freq=log_freq,
                                         log_episodes=log_episodes,
                                         verbose=False)

        elif l_type == "Imitation-SMDP-Q-Learning":
            hist, er_buffer = smdp_q_learning(env, agent, num_episodes,
                                              max_steps, **params,
                                              log_freq=log_freq,
                                              log_episodes=log_episodes,
                                              verbose=False)

        elif l_type == "Transfer-SMDP-Q-Learning":
            hist, er_buffer = smdp_q_learning(env, agent, num_episodes,
                                              max_steps, **params,
                                              log_freq=log_freq,
                                              log_episodes=log_episodes,
                                              verbose=False)

        elif l_type == "Online-SMDP-Q-Learning":
            hist = smdp_q_online_learning(env, **params,
                                          max_steps=max_steps,
                                          log_freq=log_freq,
                                          log_episodes=log_episodes,
                                          verbose=False)


        # Process results and append
        its_t, steps_t, sd_steps_t, rew_t, sd_rew_t = (hist[:, 0], hist[:, 1],
                                                       hist[:, 2], hist[:, 3],
                                                       hist[:, 4])
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
    med_steps = np.percentile(np.array(steps), 50, axis=0)
    p10_steps = np.percentile(np.array(steps), 10, axis=0)
    p90_steps = np.percentile(np.array(steps), 90, axis=0)
    # np.array(sd_steps).mean(axis=0)
    med_rew = np.percentile(np.array(rew), 50, axis=0)
    p10_rew = np.percentile(np.array(rew), 10, axis=0)
    p90_rew = np.percentile(np.array(rew), 90, axis=0)

    if save_fname is not None:
        out = np.array([its,
                        med_steps, p10_steps, p90_steps,
                        med_rew, p10_rew, p90_rew])
        np.savetxt(save_fname, out.T)
        print("Outfiled the results to {}.".format(save_fname))

    stats = {"iterations": its,
             "mean_steps": steps,
             "sd_steps": sd_steps,
             "mean_rew": rew,
             "sd_rew": sd_rew}
    return stats
