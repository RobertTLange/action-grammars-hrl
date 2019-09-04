import time
import math
import numpy as np
import pandas as pd

import gym
import gym_hanoi

from utils.q_helpers_towers import  q_learning
from grammars.cfg_grammar import get_macros
from agents.q_agent import QTable, Agent_Q
from utils.general_towers import discounted_return, get_logging_stats
from utils.general_towers import learning_params, ReplayBuffer
from utils.general_towers import macro_step, get_macros_from_productions


def smdp_q_learning_update(GAMMA, ALPHA, LAMBDA, q_func, eligibility,
                           cur_state, action, next_state, reward, done, stp,
                           old_greedy_choice=None, old_action=None, old_state=None):
    if done:
        target = discounted_return(reward, GAMMA)
    else:
        target = discounted_return(reward, GAMMA) + GAMMA**len(reward) * np.max(q_func(next_state))

    if stp > 0:
        if old_greedy_choice == action:
            eligibility(old_state)[old_action] *= GAMMA*LAMBDA
        else:
            eligibility(old_state)[old_action] = 0

    eligibility(cur_state)[action] += 1

    td_err = target - q_func(cur_state)[action]
    Q_new = q_func.table + ALPHA * td_err * eligibility.table

    q_func.update_all(Q_new)
    return eligibility, td_err


def smdp_q_learning(agent, N_DISKS, NUM_UPDATES, MAX_STEPS,
                    GAMMA, ALPHA, LAMBDA, EPSILON,
                    ROLLOUT_EVERY, NUM_ROLLOUTS, STATS_FNAME, PRINT_EVERY,
                    VERBOSE):

    start = time.time()
    log_template = "E {:>2} | T {:.1f} | Median R {:.1f} | Mean R {:.1f} | Median S {:.1f} | Mean S {:.1f}"

    reward_stats = pd.DataFrame(columns=["opt_counter", "rew_mean", "rew_sd",
                                         "rew_median", "rew_10th_p", "rew_90th_p"])

    step_stats = pd.DataFrame(columns=["opt_counter", "steps_mean", "steps_sd",
                                       "steps_median", "steps_10th_p", "steps_90th_p"])

    # Init Replay Buffer
    er_buffer = ReplayBuffer(NUM_UPDATES*MAX_STEPS, record_macros=True)

    update_counter = 0
    ep_id = 0

    env = gym.make("Hanoi-v0")
    env.set_env_parameters(N_DISKS, env_noise=0, verbose=False)

    while update_counter < NUM_UPDATES:

        state = env.reset()
        eligibility = QTable(np.zeros(N_DISKS*(3, ) + (6 + len(agent.macros),)))

        old_greedy_choice = None
        old_action = None
        old_state = None

        ep_id += 1

        for i in range(MAX_STEPS):
            action = agent.epsilon_greedy_action(state, EPSILON)
            if action > 5:
                next_state, reward, done, _ = macro_step(action, state, agent,
                                                         env, er_buffer,
                                                         ep_id)
            else:
                next_state, reward, done, _ = env.step(action)
                er_buffer.push(ep_id, state, action, reward, next_state, done, None)

            if type(reward) != list:
                reward = [reward]
            greedy_choice = agent.greedy_action(next_state)

            # Update value function
            eligibility, tde = smdp_q_learning_update(GAMMA, ALPHA, LAMBDA,
                                                      agent.q_func,  eligibility,
                                                      state, action, next_state,
                                                      reward, done, i,
                                                      old_greedy_choice, old_action,
                                                      old_state)
            update_counter += 1

            # Update variables
            old_state = state
            old_action = action
            old_greedy_choice = greedy_choice
            state = next_state

            if done:
                break

            if (update_counter+1) % ROLLOUT_EVERY == 0:
                r_stats, s_stats = get_logging_stats(update_counter, agent, GAMMA,
                                                     N_DISKS, MAX_STEPS, NUM_ROLLOUTS)

                reward_stats = pd.concat([reward_stats, r_stats], axis=0)
                step_stats = pd.concat([step_stats, s_stats], axis=0)

            if VERBOSE and update_counter % PRINT_EVERY == 0:
                stop = time.time()
                print(log_template.format(update_counter, stop-start,
                                          r_stats.loc[0, "rew_median"],
                                          r_stats.loc[0, "rew_mean"],
                                          s_stats.loc[0, "steps_median"],
                                          s_stats.loc[0, "steps_mean"]))
                start = time.time()

    # Save the logging dataframe
    df_to_save = pd.concat([reward_stats, step_stats], axis=1)
    df_to_save = df_to_save.loc[:, ~df_to_save.columns.duplicated()]
    df_to_save = df_to_save.reset_index()
    df_to_save.to_csv("results/TOH/" + STATS_FNAME)
    return df_to_save


def smdp_q_online_learning(N_DISKS, NUM_UPDATES, MAX_STEPS,
                           GAMMA, ALPHA, LAMBDA, EPSILON,
                           ROLLOUT_EVERY, NUM_ROLLOUTS, STATS_FNAME,
                           PRINT_EVERY, VERBOSE,
                           seq_k_schedule, seq_update_schedule):
    start = time.time()
    log_template = "E {:>2} | T {:.1f} | Median R {:.1f} | Mean R {:.1f} | Median S {:.1f} | Mean S {:.1f}"

    reward_stats = pd.DataFrame(columns=["opt_counter", "rew_mean", "rew_sd",
                                         "rew_median", "rew_10th_p", "rew_90th_p"])

    step_stats = pd.DataFrame(columns=["opt_counter", "steps_mean", "steps_sd",
                                       "steps_median", "steps_10th_p", "steps_90th_p"])

    # Init Replay Buffer
    er_buffer = ReplayBuffer(NUM_UPDATES*MAX_STEPS, record_macros=True)

    update_counter = 0
    ep_id = 0

    env = gym.make("Hanoi-v0")
    env.set_env_parameters(N_DISKS, env_noise=0, verbose=False)

    while update_counter < NUM_UPDATES:

        state = env.reset()
        eligibility = QTable(np.zeros(N_DISKS*(3, ) + (6 + len(agent.macros),)))

        old_greedy_choice = None
        old_action = None
        old_state = None

        ep_id += 1

        for i in range(MAX_STEPS):
            action = agent.epsilon_greedy_action(state, EPSILON)
            if action > 5:
                next_state, reward, done, _ = macro_step(action, state, agent,
                                                         env, er_buffer,
                                                         ep_id)
            else:
                next_state, reward, done, _ = env.step(action)
                er_buffer.push(ep_id, state, action, reward, next_state, done, None)

            if type(reward) != list:
                reward = [reward]
            greedy_choice = agent.greedy_action(next_state)

            # Update value function
            eligibility, tde = smdp_q_learning_update(GAMMA, ALPHA, LAMBDA,
                                                      agent.q_func,  eligibility,
                                                      state, action, next_state,
                                                      reward, done, i,
                                                      old_greedy_choice, old_action,
                                                      old_state)
            update_counter += 1

            # Update variables
            old_state = state
            old_action = action
            old_greedy_choice = greedy_choice
            state = next_state

            if done:
                break

            if (update_counter+1) % ROLLOUT_EVERY == 0:
                r_stats, s_stats = get_logging_stats(update_counter, agent, GAMMA,
                                                     N_DISKS, MAX_STEPS, NUM_ROLLOUTS)

                reward_stats = pd.concat([reward_stats, r_stats], axis=0)
                step_stats = pd.concat([step_stats, s_stats], axis=0)

            if VERBOSE and update_counter % PRINT_EVERY == 0:
                stop = time.time()
                print(log_template.format(update_counter, stop-start,
                                          r_stats.loc[0, "rew_median"],
                                          r_stats.loc[0, "rew_mean"],
                                          s_stats.loc[0, "steps_median"],
                                          s_stats.loc[0, "steps_mean"]))
                start = time.time()

    # Save the logging dataframe
    df_to_save = pd.concat([reward_stats, step_stats], axis=1)
    df_to_save = df_to_save.loc[:, ~df_to_save.columns.duplicated()]
    df_to_save = df_to_save.reset_index()
    df_to_save.to_csv("results/TOH/" + STATS_FNAME)
    return df_to_save
