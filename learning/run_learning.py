import time
import gym
import gym_hanoi
import numpy as np
from agents import Agent_Q, SMDP_Agent_Q, Macro
from learning.q_learning import q_learning
from learning.smdp_q_learning import smdp_q_learning
from learning.learning_params import learning_parameters


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
