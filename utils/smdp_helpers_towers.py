import time
import math
import numpy as np
import pandas as pd

import gym
import gym_hanoi

from utils.q_helpers_towers import  q_learning
from utils.cfg_grammar import get_macros, letter_to_action, action_to_letter
from agents.q_agent import QTable, Agent_Q
from utils.general_towers import discounted_return, get_logging_stats
from utils.general_towers import learning_params, ReplayBuffer, rollout_episode
from utils.general_towers import macro_step, get_macros_from_productions
from utils.q_helpers_towers import q_learning_update
from agents.smdp_q_agent import SMDP_Agent_Q


def smdp_q_learning_update(GAMMA, ALPHA, LAMBDA, q_func, eligibility,
                           cur_state, action, next_state, reward, done, stp,
                           old_greedy_choice=None, old_action=None, old_state=None):
    if done:
        target = discounted_return(reward, GAMMA)
    else:
        target = discounted_return(reward, GAMMA) + GAMMA**len(reward) * np.max(q_func(next_state))

    if stp > 0 and eligibility is not None:
        if old_greedy_choice == action:
            eligibility(old_state)[old_action] *= GAMMA*LAMBDA
        else:
            eligibility(old_state)[old_action] = 0

    if eligibility is not None:
        eligibility(cur_state)[action] += 1
        Q_new = q_func.table + ALPHA * td_err * eligibility.table
        q_func.update_all(Q_new)
    else:
        td_err = target - q_func(cur_state)[action]
        q_updated = q_func(cur_state)[action] + ALPHA * td_err
        q_func.update_table(cur_state, q_updated, action)
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
        step_counter = 0

        while step_counter < MAX_STEPS:
            action = agent.epsilon_greedy_action(state, EPSILON)
            if action > 5:
                next_state, reward, done, er_temp = macro_step(action, state, agent,
                                                         env, er_buffer,
                                                         ep_id)
                steps = agent.macros[action - 6].macro_len
            else:
                next_state, reward, done, _ = env.step(action)
                er_buffer.push(ep_id, state, action, reward, next_state, done, None)
                steps = 1

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
            step_counter += steps
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
                           PRINT_EVERY, VERBOSE, GRAMMAR_DIR, g_type,
                           seq_k_schedule, seq_update_schedule):
    start = time.time()
    log_template = "E {:>2} | T {:.1f} | Median R {:.1f} | Mean R {:.1f} | Median S {:.1f} | Mean S {:.1f}"

    reward_stats = pd.DataFrame(columns=["opt_counter", "rew_mean", "rew_sd",
                                         "rew_median", "rew_10th_p", "rew_90th_p"])

    step_stats = pd.DataFrame(columns=["opt_counter", "steps_mean", "steps_sd",
                                       "steps_median", "steps_10th_p", "steps_90th_p"])

    update_counter = 0
    ep_id = 0

    # Initialize counter to keep track of grammar updates
    grammar_counter = 0
    env = gym.make("Hanoi-v0")
    env.set_env_parameters(N_DISKS, env_noise=0, verbose=False)

    # Init Replay Buffer
    er_buffer = ReplayBuffer(NUM_UPDATES*MAX_STEPS, record_macros=True)
    # Init Value Transfer Buffer
    value_buffer = TransferValueBuffer(env)

    # Run initial Q-Learning before first macro acquisition
    agent = Agent_Q(env)
    while update_counter < seq_update_schedule[0]:
        state = env.reset()

        old_greedy_choice = None
        old_action = None
        old_state = None

        for i in range(MAX_STEPS):
            action = agent.epsilon_greedy_action(state, EPSILON)
            next_state, reward, done, _ = env.step(action)
            greedy_choice = agent.greedy_action(next_state)

            # Update value function
            eligibility, tde = q_learning_update(GAMMA, ALPHA, LAMBDA,
                                                 agent.q_func, None,
                                                 state, action, next_state,
                                                 reward, done, i,
                                                 old_greedy_choice, old_action,
                                                 old_state)
            update_counter += 1
            # Extend replay buffer
            er_buffer.push(ep_id, state, action, reward, next_state, done)

            # Update variables
            old_state = state
            old_action = action
            old_greedy_choice = greedy_choice
            state = next_state

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

            if done:
                break

        ep_id += 1

    # Rollout Episode with agent and infer macros
    _, _, buffer_to_infer = rollout_episode(agent, MAX_STEPS, N_DISKS, GAMMA,
                                            record_macros=False)

    SENTENCE = []
    for step in range(len(buffer_to_infer)):
        SENTENCE.append(action_to_letter(buffer_to_infer[step][2]))

    SENTENCE = "".join(SENTENCE)
    NUM_MACROS = 5
    macros, counts, stats = get_macros(NUM_MACROS, SENTENCE, 6, GRAMMAR_DIR,
                                k=seq_k_schedule[0], g_type=g_type)
    print(macros)
    agent = endorse_agent(env, agent, macros, value_buffer)
    grammar_counter += 1

    compression_stats = [stats]
    # Run SMDP-Loop with Inferred Grammar Macros
    while update_counter < NUM_UPDATES:
        state = env.reset()

        old_greedy_choice = None
        old_action = None
        old_state = None

        ep_id += 1

        for i in range(MAX_STEPS):
            # CHECK AND PERFORM GRAMMAR UPDATE!
            if update_counter == seq_update_schedule[grammar_counter]:
                # Rollout Episode with agent and infer macros
                _, _, buffer_to_infer = rollout_episode(agent, MAX_STEPS, N_DISKS, GAMMA,
                                                        record_macros=True)

                SENTENCE = []
                for step in range(len(buffer_to_infer)):
                    SENTENCE.append(action_to_letter(buffer_to_infer[step][2]))

                SENTENCE = "".join(SENTENCE)
                MAX_NUM_MACROS = 5
                macros, counts, stats = get_macros(NUM_MACROS, SENTENCE, 6, GRAMMAR_DIR,
                                                   k=seq_k_schedule[grammar_counter], g_type=g_type)

                compression_stats.append(stats)
                print(macros)
                agent = endorse_agent(env, agent, macros, value_buffer)
                if grammar_counter < len(seq_update_schedule)-1:
                    grammar_counter += 1

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
            elig, tde = smdp_q_learning_update(GAMMA, ALPHA, LAMBDA,
                                                      agent.q_func,  None,
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

    # Save the compression stas
    compression_array = np.array(compression_stats)
    np.savetxt("results/TOH/compression_" + STATS_FNAME, compression_array)
    # Save the logging dataframe
    df_to_save = pd.concat([reward_stats, step_stats], axis=1)
    df_to_save = df_to_save.loc[:, ~df_to_save.columns.duplicated()]
    df_to_save = df_to_save.reset_index()
    df_to_save.to_csv("results/TOH/" + STATS_FNAME)
    return df_to_save


class TransferValueBuffer():
    def __init__(self, env):
        self.primitive_values = None
        self.macro_values = {}
        self.env = env
        self.num_primitives = 6
        self.macros_active = None

    def store_primitive_values(self, agent_old):
        self.primitive_values = agent_old.q_func.table[..., :self.num_primitives]
        return

    def store_macro_values(self, agent_old):
        if self.macros_active is not None:
            for i, macro in enumerate(self.macros_active):
                self.macro_values[macro] = agent_old.q_func.table[..., self.num_primitives+i]

    def transfer_values(self, macros_to_transfer):
        # Initialize an agent for the new set of macros
        macros_to_transfer_temp = get_macros_from_productions(macros_to_transfer)
        agent = SMDP_Agent_Q(self.env, macros_to_transfer_temp)
        # Perform the value transfer for primitive values
        agent.q_func.table[..., :self.num_primitives] = self.primitive_values
        # Perform the value transfer for macro values
        for i, macro in enumerate(macros_to_transfer):
            if macro in self.macro_values.keys():
                agent.q_func.table[..., self.num_primitives+i] = self.macro_values[macro]
        # Update currently active set of macro actions
        self.macros_active = macros_to_transfer
        return agent


def endorse_agent(env, agent_old, macros_to_transfer, value_buffer):
    # Store the primitive action values in the value buffer - later transfer
    value_buffer.store_primitive_values(agent_old)
    # Store the current macro values in the value buffer - become inactive
    value_buffer.store_macro_values(agent_old)
    # Transfer the values over to the new agent
    transfer_agent = value_buffer.transfer_values(macros_to_transfer)
    return transfer_agent

# python run_learning_towers.py --N_DISKS 5 --LEARN_TYPE Online-SMDP-Q-Learning --RUN_TIMES 1 --GRAMMAR_TYPE G-Lexis --SAVE_FNAME seq_TOH.csv --VERBOSE
