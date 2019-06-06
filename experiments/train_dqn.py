import gym
import time
import numpy as np
import pandas as pd
import gridworld

import torch
import torch.autograd as autograd

from dqn import MLP_DQN, MLP_DDQN, init_agent
from dqn_helpers import command_line_dqn, ReplayBuffer, update_target, epsilon_by_episode, compute_td_loss, get_logging_stats


def run_dqn_learning(args):
    log_template = "E {:>2} | T {:.1f} | Median R {:.1f} | Mean R {:.1f} | Median S {:.1f} | Mean S {:.1f}"

    # Set the GPU device on which to run the agent
    USE_CUDA = torch.cuda.is_available()
    if USE_CUDA:
        torch.cuda.set_device(args.device_id)
        print("USING CUDA DEVICE {}".format(args.device_id))
    else:
        print("USING CPU")
    Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)
    start = time.time()

    # Extract variables for arguments
    TRAIN_BATCH_SIZE = args.TRAIN_BATCH_SIZE
    EPS_START, EPS_STOP, EPS_DECAY = args.EPS_START, args.EPS_STOP, args.EPS_DECAY
    GAMMA, L_RATE = args.GAMMA, args.L_RATE

    NUM_EPISODES = args.NUM_EPISODES
    NUM_ROLLOUTS = args.NUM_ROLLOUTS
    MAX_STEPS = args.MAX_STEPS
    ROLLOUT_EVERY = args.ROLLOUT_EVERY
    SAVE_EVERY = args.SAVE_EVERY
    UPDATE_EVERY = args.UPDATE_EVERY
    PRINT_EVERY = args.PRINT_EVERY
    VERBOSE = args.VERBOSE

    AGENT = args.AGENT
    AGENT_FNAME = args.AGENT_FNAME
    STATS_FNAME = args.STATS_FNAME

    # Setup agent, replay replay_buffer, logging stats df
    if AGENT == "MLP-DQN":
        agents, optimizer = init_agent(MLP_DQN, L_RATE, USE_CUDA)
    elif AGENT == "MLP-Dueling-DQN":
        agents, optimizer = init_agent(MLP_DDQN, L_RATE, USE_CUDA)

    replay_buffer = ReplayBuffer(capacity=5000)

    reward_stats = pd.DataFrame(columns=["opt_counter", "rew_mean", "rew_sd",
                                         "rew_median", "rew_10th_p", "rew_90th_p"])

    step_stats = pd.DataFrame(columns=["opt_counter", "steps_mean", "steps_sd",
                                       "steps_median", "steps_10th_p", "steps_90th_p"])

    # Initialize optimization update counter and environment
    opt_counter = 0
    env = gym.make("dense-v0")

    # RUN TRAINING LOOP OVER EPISODES
    for ep_id in range(NUM_EPISODES):
        epsilon = epsilon_by_episode(ep_id + 1, EPS_START, EPS_STOP, EPS_DECAY)

        obs = env.reset()

        steps = 0
        while steps < MAX_STEPS:
            action = agents["current"].act(obs.flatten(), epsilon)
            next_obs, rew, done, _  = env.step(action)
            steps += 1

            # Push transition to ER Buffer
            replay_buffer.push(ep_id, steps, obs, action,
                               rew, next_obs, done)

            if len(replay_buffer) > TRAIN_BATCH_SIZE:
                opt_counter += 1
                loss = compute_td_loss(agents, optimizer, replay_buffer,
                                       TRAIN_BATCH_SIZE, GAMMA, Variable)


            # Go to next episode if current one terminated or update obs
            if done: break
            else: obs = next_obs

            # On-Policy Rollout for Performance evaluation
            if (opt_counter+1) % ROLLOUT_EVERY == 0:
                r_stats, s_stats = get_logging_stats(opt_counter, agents,
                                                     GAMMA, NUM_ROLLOUTS, MAX_STEPS)
                reward_stats = pd.concat([reward_stats, r_stats], axis=0)
                step_stats = pd.concat([step_stats, s_stats], axis=0)

            if (opt_counter+1) % UPDATE_EVERY == 0:
                update_target(agents["current"], agents["target"])

            if (opt_counter+1) % SAVE_EVERY == 0:
                # Save the model checkpoint - for single "representative agent"
                torch.save(agents["current"].state_dict(), AGENT_FNAME)
                # Save the logging dataframe
                df_to_save = pd.concat([reward_stats, step_stats], axis=1)
                df_to_save = df_to_save.loc[:,~df_to_save.columns.duplicated()]
                df_to_save.to_csv(STATS_FNAME)

        if VERBOSE and (ep_id+1) % PRINT_EVERY == 0:
            stop = time.time()
            print(log_template.format(ep_id+1, stop-start,
                                      r_stats.loc[0, "rew_median"],
                                      r_stats.loc[0, "rew_mean"],
                                      s_stats.loc[0, "steps_median"],
                                      s_stats.loc[0, "steps_mean"]))
            start = time.time()

    # Finally save all results!
    torch.save(agents["current"].state_dict(), "agents/" + AGENT_FNAME)
    # Save the logging dataframe
    df_to_save = pd.concat([reward_stats, step_stats], axis=1)
    df_to_save = df_to_save.loc[:,~df_to_save.columns.duplicated()]
    df_to_save = df_to_save.reset_index()
    df_to_save.to_csv("results/"  + args.AGENT + "_" + STATS_FNAME)
    return df_to_save


def run_multiple_times(args, run_fct):

    df_across_runs = []
    print("START RUNNING {} AGENT LEARNING FOR {} TIMES".format(args.AGENT,
                                                                args.RUN_TIMES))
    for t in range(args.RUN_TIMES):
        start_t = time.time()
        df_temp = run_fct(args)
        df_across_runs.append(df_temp)
        total_t = time.time() - start_t
        print("Done training {}/{} runs after {:.2f} Secs".format(t+1,
                                                                  args.RUN_TIMES,
                                                                  total_t))

    df_concat = pd.concat(df_across_runs)
    by_row_index = df_concat.groupby(df_concat.index)
    df_means = by_row_index.mean()
    df_means.to_csv("results/" + str(args.RUN_TIMES) + "_RUNS_" + args.AGENT + "_" + args.STATS_FNAME)
    return df_means


if __name__ == "__main__":
    args = command_line_dqn()

    if args.RUN_TIMES == 1:
        print("START RUNNING {} AGENT LEARNING FOR 1 TIME".format(args.AGENT))
        run_dqn_learning(args)
    else:
        run_multiple_times(args, run_dqn_learning)
