import gym
import time
import numpy as np
import pandas as pd
import gridworld

import torch
import torch.autograd as autograd

from dueling_dqn import MLP_DDQN, init_agent
from dqn_helpers import command_line_dqn, ReplayBuffer, update_target, epsilon_by_episode, compute_td_loss, get_logging_stats


def main(args):
    log_template = "E {:>2} | T {:.1f} | Median R {:.1f} | Mean R {:.1f} | Median S {:.1f} | Mean S {:.1f}"

    # Set the GPU device on which to run the agent
    USE_CUDA = torch.cuda.is_available()
    if USE_CUDA: torch.cuda.set_device(args.device_id)
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
    VERBOSE = args.VERBOSE

    AGENT_FNAME = args.AGENT_FNAME
    STATS_FNAME = args.STATS_FNAME

    # Setup agent, replay replay_buffer, logging stats df
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
        episode_rew = 0

        for j in range(MAX_STEPS):
            action = agents["current"].act(obs.flatten(), epsilon)
            next_obs, rew, done, _  = env.step(action)
            episode_rew += rew

            # Push transition to ER Buffer
            replay_buffer.push(ep_id, j+1, obs, action,
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

                if VERBOSE:
                    stop = time.time()
                    print(log_template.format(ep_id, stop-start,
                                              r_stats.loc[0, "rew_median"],
                                              r_stats.loc[0, "rew_mean"],
                                              s_stats.loc[0, "steps_median"],
                                              s_stats.loc[0, "steps_mean"]))
                    start = time.time()

            if (opt_counter+1) % UPDATE_EVERY == 0:
                update_target(agents["current"], agents["target"])

            if (opt_counter+1) % SAVE_EVERY == 0:
                # Save the model checkpoint - for single "representative agent"
                torch.save(agents["current"].state_dict(), AGENT_FNAME)
                # Save the logging dataframe
                df_to_save = pd.concat([reward_stats, step_stats], axis=1)
                df_to_save = df_to_save.loc[:,~df_to_save.columns.duplicated()]
                df_to_save.to_csv(STATS_FNAME)

    return


if __name__ == "__main__":
    args = command_line_dqn()
    main(args)
