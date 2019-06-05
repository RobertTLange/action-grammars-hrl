import gym
import time
import numpy as np
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

    # Setup agent and replay replay_buffer
    agents, optimizer = init_agent(MLP_DDQN, L_RATE, USE_CUDA)
    replay_buffer = ReplayBuffer(capacity=5000)

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

            if done: break
            else: obs = next_obs

        # On-Policy Rollout for Performance evaluation
        if ep_id % ROLLOUT_EVERY == 0:
            r_stats, s_stats = get_logging_stats(env, agents, GAMMA, NUM_ROLLOUTS, MAX_STEPS)

            if VERBOSE:
                stop = time.time()
                print(log_template.format(ep_id, stop-start,
                                           r_stats["median"], r_stats["mean"],
                                           s_stats["median"], s_stats["mean"]))
                start = time.time()

        if ep_id % UPDATE_EVERY == 0:
            update_target(agents["current"], agents["target"])

        if ep_id % SAVE_EVERY == 0:
            # Save the model checkpoint - for single "representative agent"
            torch.save(agents["current"].state_dict(), AGENT_FNAME)

    return


if __name__ == "__main__":
    args = command_line_dqn()
    main(args)
