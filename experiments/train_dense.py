import gym
import matplotlib.pyplot as plt
import numpy as np
import gridworld

from dueling_dqn import MLP_DDQN, init_agent
from dqn_helpers import command_line_dqn, ReplayBuffer, update_target, epsilon_by_episode, compute_td_loss


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
    MAX_STEPS = args.MAX_STEPS
    ROLLOUT_EVERY = args.ROLLOUT_EVERY
    SAVE_EVERY = args.SAVE_EVERY
    UPDATE_EVERY = args.UPDATE_EVERY

    # Setup agent and replay replay_buffer
    agents, optimizer = init_agent(MLP_DDQN, args)
    replay_buffer = ReplayBuffer()

    # Initialize optimization update counter and environment
    opt_counter = 0
    env = gym.make("dense-v0")

    # RUN TRAINING LOOP OVER EPISODES
    for ep_id in range(NUM_EPISODES):
        epsilon = epsilon_by_episode(ep_id + 1, EPS_START, EPS_STOP, EPS_DECAY)

        obs = env.reset(capacity=5000)
        done = False
        episode_rew = 0

        steps = 0
        for j in range(MAX_STEPS):
            action = agent["current"].act(obs, epsilon)
            next_obs, rew, done, _  = env.step(action)
            steps += 1
            episode_rew += rew

            if done:
                break

            replay_buffer.push(ep_id, steps, obs, action,
                               rew, next_obs, done)

            obs = next_obs

            if len(replay_buffer) > TRAIN_BATCH_SIZE:
                opt_counter += 1
                loss = compute_td_loss(agents, optimizer, replay_buffer,
                                       args, Variable)

        # On-Policy Rollout for Performance evaluation
        if ep_id % ROLLOUT_EVERY == 0:
            reward_stats, steps_stats, buffer = get_logging_stats(env,
                                                                  agents,
                                                                  params)

            logger.update_performance(ep_id, reward_stats, steps_stats)

            if verbose:
                stop = time.time()
                print(log_template.format(ep_id, stop-start,
                                           reward_stats["sum_median"], reward_stats["sum_mean"],
                                           steps_stats["median"], steps_stats["mean"]))
                start = time.time()

        if ep_id % UPDATE_EVERY == 0:
            update_target(agents["current"], agents["target"])
            # Save the model checkpoint - for single "representative agent"
            torch.save(agents["current"].state_dict(), params.agent_fname)

    return log


if __name__ == "__main__":
    args = command_line_dqn()
    main(args)
