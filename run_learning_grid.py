import gym
import time
import numpy as np
import pandas as pd
import gridworld

import torch
import torch.autograd as autograd
import torch.multiprocessing as mp

from agents.dqn import CNN_DDQN, MLP_DQN, MLP_DDQN, init_agent
from utils.general_dqn import command_line_dqn_grid, ReplayBuffer, update_target, epsilon_by_episode
from utils.general_dqn import compute_td_loss, get_logging_stats, run_multiple_times
from utils.smdp_helpers_dqn import MacroBuffer, macro_action_exec, get_macro_from_agent
from utils.smdp_helpers_dqn import command_line_grammar_dqn
from utils.atari_wrapper import make_atari, wrap_deepmind, wrap_pytorch

SEQ_DIR = "grammars/sequitur/"
LEXIS_DIR = "grammars/Lexis/"

log_template = "Step {:>2} | T {:.1f} | Median R {:.1f} | Mean R {:.1f} | Median S {:.1f} | Mean S {:.1f}"

def run_dqn_learning(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
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

    NUM_UPDATES = args.NUM_UPDATES
    NUM_ROLLOUTS = args.NUM_ROLLOUTS
    MAX_STEPS = args.MAX_STEPS
    ROLLOUT_EVERY = args.ROLLOUT_EVERY
    UPDATE_EVERY = args.UPDATE_EVERY
    VERBOSE = args.VERBOSE
    PRINT_EVERY = args.PRINT_EVERY
    CAPACITY = args.CAPACITY

    ENV_ID = args.ENV_ID
    AGENT = args.AGENT
    AGENT_FNAME = args.AGENT_FNAME
    STATS_FNAME = args.SAVE_FNAME

    if args.DOUBLE: TRAIN_DOUBLE = True
    else: TRAIN_DOUBLE = False

    # Setup agent, replay replay_buffer, logging stats df
    if AGENT == "MLP-DQN" or AGENT == "DOUBLE":
        agents, optimizer = init_agent(MLP_DQN, L_RATE, USE_CUDA)
    elif AGENT == "MLP-Dueling-DQN":
        agents, optimizer = init_agent(MLP_DDQN, L_RATE, USE_CUDA)
    elif AGENT == "CNN-Dueling-DQN":
        agents, optimizer = init_agent(CNN_DDQN, L_RATE, USE_CUDA)

    replay_buffer = ReplayBuffer(capacity=CAPACITY)

    reward_stats = pd.DataFrame(columns=["opt_counter", "rew_mean", "rew_sd",
                                         "rew_median", "rew_10th_p", "rew_90th_p"])

    step_stats = pd.DataFrame(columns=["opt_counter", "steps_mean", "steps_sd",
                                       "steps_median", "steps_10th_p", "steps_90th_p"])

    # Initialize optimization update counter and environment
    opt_counter = 0
    if ENV_ID == "dense-v0":
        env = gym.make(ENV_ID)
    else:
        # Wrap the ATARI env in DeepMind Wrapper
        env = make_atari(ENV_ID)
        env = wrap_deepmind(env, episode_life=True, clip_rewards=True,
                            frame_stack=True, scale=True)
        env = wrap_pytorch(env)

    # RUN TRAINING LOOP OVER EPISODES
    while opt_counter < NUM_UPDATES:
        epsilon = epsilon_by_episode(opt_counter + 1, EPS_START, EPS_STOP, EPS_DECAY)

        obs = env.reset()
        ep_id = 0
        steps = 0
        while steps < MAX_STEPS:
            if ENV_ID == "dense-v0":
                action = agents["current"].act(obs.flatten(), epsilon)
            else:
                action = agents["current"].act(obs, epsilon)
            next_obs, rew, done, _  = env.step(action)
            steps += 1

            # Push transition to ER Buffer
            replay_buffer.push(ep_id, steps, obs, action,
                               rew, next_obs, done)

            if len(replay_buffer) > TRAIN_BATCH_SIZE:
                opt_counter += 1
                loss = compute_td_loss(agents, optimizer, replay_buffer,
                                       TRAIN_BATCH_SIZE, GAMMA, Variable,
                                       TRAIN_DOUBLE, ENV_ID)


            # On-Policy Rollout for Performance evaluation
            if (opt_counter+1) % ROLLOUT_EVERY == 0:
                r_stats, s_stats = get_logging_stats(opt_counter, agents,
                                                     GAMMA, NUM_ROLLOUTS,
                                                     MAX_STEPS, ENV_ID)
                reward_stats = pd.concat([reward_stats, r_stats], axis=0)
                step_stats = pd.concat([step_stats, s_stats], axis=0)

            if (opt_counter+1) % UPDATE_EVERY == 0:
                update_target(agents["current"], agents["target"])

            if VERBOSE and (opt_counter+1) % PRINT_EVERY == 0:
                stop = time.time()
                print(log_template.format(opt_counter+1, stop-start,
                                          r_stats.loc[0, "rew_median"],
                                          r_stats.loc[0, "rew_mean"],
                                          s_stats.loc[0, "steps_median"],
                                          s_stats.loc[0, "steps_mean"]))
                start = time.time()

            if args.SAVE:
                if ENV_ID == "dense-v0":
                    # Gridworld - Updates after which to save expert/transfer agent
                    save_after_upd = [250000, 500000, 1000000]
                else:
                    # ATARI - Updates after which to save expert/transfer agent
                    save_after_upd = [1000000, 2500000, 5000000]
                if opt_counter+1 in save_after_upd:
                    agent_path = "agents/trained/" + str(opt_counter+1) + "_" + AGENT_FNAME
                    torch.save(agents["current"].state_dict(), agent_path)
                    print("Saved expert agent to {}".format(agent_path))

            # Go to next episode if current one terminated or update obs
            if done: break
            else: obs = next_obs
        ep_id += 1

    # Save the logging dataframe
    df_to_save = pd.concat([reward_stats, step_stats], axis=1)
    df_to_save = df_to_save.loc[:,~df_to_save.columns.duplicated()]
    df_to_save = df_to_save.reset_index()

    if args.SAVE:
        # Finally save all results!
        torch.save(agents["current"].state_dict(), "agents/" + str(NUM_UPDATES) + "_" + AGENT_FNAME)
        df_to_save.to_csv("results/"  + args.AGENT + "_" + STATS_FNAME)
    return df_to_save


def run_smdp_dqn_learning(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

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

    NUM_UPDATES = args.NUM_UPDATES
    NUM_ROLLOUTS = args.NUM_ROLLOUTS
    MAX_STEPS = args.MAX_STEPS
    ROLLOUT_EVERY = args.ROLLOUT_EVERY
    UPDATE_EVERY = args.UPDATE_EVERY
    VERBOSE = args.VERBOSE
    PRINT_EVERY = args.PRINT_EVERY
    CAPACITY = args.CAPACITY

    ENV_ID = args.ENV_ID
    AGENT = args.AGENT
    AGENT_FNAME = args.AGENT_FNAME
    STATS_FNAME = args.SAVE_FNAME

    # Get macros from expert dqn rollout
    LOAD_CKPT = args.LOAD_CKPT
    NUM_MACROS = args.NUM_MACROS
    GRAMMAR_TYPE = args.GRAMMAR_TYPE

    if GRAMMAR_TYPE == "sequitur":
        GRAMMAR_DIR = SEQ_DIR
    elif GRAMMAR_TYPE == "lexis":
        GRAMMAR_DIR = LEXIS_DIR

    macros, counts = get_macro_from_agent(NUM_MACROS, 4, USE_CUDA, AGENT,
                                          LOAD_CKPT, GRAMMAR_DIR, ENV_ID, g_type=GRAMMAR_TYPE)

    NUM_ACTIONS = 4 + NUM_MACROS
    if AGENT == "DOUBLE": TRAIN_DOUBLE = True
    else: TRAIN_DOUBLE = False

    # Setup agent, replay replay_buffer, logging stats df
    if AGENT == "MLP-DQN" or AGENT == "DOUBLE":
        agents, optimizer = init_agent(MLP_DQN, L_RATE, USE_CUDA, NUM_ACTIONS)
    elif AGENT == "MLP-Dueling-DQN":
        agents, optimizer = init_agent(MLP_DDQN, L_RATE, USE_CUDA, NUM_ACTIONS)
    elif AGENT == "CNN-Dueling-DQN":
        agents, optimizer = init_agent(CNN_DDQN, L_RATE, USE_CUDA, NUM_ACTIONS)

    replay_buffer = ReplayBuffer(capacity=5000)
    reward_stats = pd.DataFrame(columns=["opt_counter", "rew_mean", "rew_sd",
                                         "rew_median", "rew_10th_p", "rew_90th_p"])

    step_stats = pd.DataFrame(columns=["opt_counter", "steps_mean", "steps_sd",
                                       "steps_median", "steps_10th_p", "steps_90th_p"])

    # Initialize optimization update counter and environment
    opt_counter = 0
    if ENV_ID == "dense-v0":
        env = gym.make(ENV_ID)
    else:
        # Wrap the ATARI env in DeepMind Wrapper
        env = make_atari(ENV_ID)
        env = wrap_deepmind(env, episode_life=True, clip_rewards=True,
                            frame_stack=True, scale=True)
        env = wrap_pytorch(env)

    ep_id = 0
    # RUN TRAINING LOOP OVER EPISODES
    while opt_counter < NUM_UPDATES:
        epsilon = epsilon_by_episode(ep_id + 1, EPS_START, EPS_STOP, EPS_DECAY)

        obs = env.reset()

        steps = 0
        while steps < MAX_STEPS:
            if ENV_ID == "dense-v0":
                action = agents["current"].act(obs.flatten(), epsilon)
            else:
                action = agents["current"].act(obs, epsilon)

            if action < 4:
                next_obs, rew, done, _  = env.step(action)
                steps += 1

                # Push transition to ER Buffer
                replay_buffer.push(ep_id, steps, obs, action,
                                   rew, next_obs, done)
            else:
                # Need to execute a macro action
                macro = macros[action - 4]
                next_obs, macro_rew, done, _ = macro_action_exec(ep_id, obs,
                                                                 steps,
                                                                 replay_buffer,
                                                                 macro, env,
                                                                 GAMMA)
                steps += len(macro)

            if len(replay_buffer) > TRAIN_BATCH_SIZE:
                opt_counter += 1
                loss = compute_td_loss(agents, optimizer, replay_buffer,
                                       TRAIN_BATCH_SIZE, GAMMA, Variable,
                                       TRAIN_DOUBLE, ENV_ID)

            ep_id += 1
            # On-Policy Rollout for Performance evaluation
            if (opt_counter+1) % ROLLOUT_EVERY == 0:
                r_stats, s_stats = get_logging_stats(opt_counter, agents,
                                                     GAMMA, NUM_ROLLOUTS,
                                                     MAX_STEPS, ENV_ID)
                reward_stats = pd.concat([reward_stats, r_stats], axis=0)
                step_stats = pd.concat([step_stats, s_stats], axis=0)

            if (opt_counter+1) % UPDATE_EVERY == 0:
                update_target(agents["current"], agents["target"])

            if VERBOSE and (opt_counter+1) % PRINT_EVERY == 0:
                stop = time.time()
                print(log_template.format(opt_counter+1, stop-start,
                                          r_stats.loc[0, "rew_median"],
                                          r_stats.loc[0, "rew_mean"],
                                          s_stats.loc[0, "steps_median"],
                                          s_stats.loc[0, "steps_mean"]))
                start = time.time()
            # Go to next episode if current one terminated or update obs
            if done: break
            else: obs = next_obs

        ep_id +=1

    # Save the logging dataframe
    df_to_save = pd.concat([reward_stats, step_stats], axis=1)
    df_to_save = df_to_save.loc[:,~df_to_save.columns.duplicated()]
    df_to_save = df_to_save.reset_index()

    if args.SAVE:
        # Finally save all results!
        torch.save(agents["current"].state_dict(),
                   "agents/" + AGENT + "_" + AGENT_FNAME)
        df_to_save.to_csv("results/" + str(NUM_MACROS) + "_" + STATS_FNAME)
    return df_to_save


def run_online_dqn_smdp_learning(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

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

    NUM_UPDATES = args.NUM_UPDATES
    NUM_ROLLOUTS = args.NUM_ROLLOUTS
    MAX_STEPS = args.MAX_STEPS
    ROLLOUT_EVERY = args.ROLLOUT_EVERY
    UPDATE_EVERY = args.UPDATE_EVERY
    VERBOSE = args.VERBOSE
    PRINT_EVERY = args.PRINT_EVERY
    CAPACITY = args.CAPACITY

    ENV_ID = args.ENV_ID
    AGENT = args.AGENT
    AGENT_FNAME = args.AGENT_FNAME
    STATS_FNAME = args.SAVE_FNAME

    # Get macros from expert dqn rollout
    LOAD_CKPT = args.LOAD_CKPT
    NUM_MACROS = args.NUM_MACROS
    GRAMMAR_TYPE = args.GRAMMAR_TYPE
    GRAMMAR_EVERY = args.GRAMMAR_EVERY

    if GRAMMAR_TYPE == "sequitur":
        GRAMMAR_DIR = SEQ_DIR
    elif GRAMMAR_TYPE == "lexis":
        GRAMMAR_DIR = LEXIS_DIR

    NUM_ACTIONS = 4 + NUM_MACROS
    if AGENT == "DOUBLE": TRAIN_DOUBLE = True
    else: TRAIN_DOUBLE = False

    # Setup agent, replay replay_buffer, logging stats df
    if AGENT == "MLP-DQN" or AGENT == "DOUBLE":
        agents, optimizer = init_agent(MLP_DQN, L_RATE, USE_CUDA)
    elif AGENT == "MLP-Dueling-DQN":
        agents, optimizer = init_agent(MLP_DDQN, L_RATE, USE_CUDA)
    elif AGENT == "CNN-Dueling-DQN":
        agents, optimizer = init_agent(CNN_DDQN, L_RATE, USE_CUDA)

    # Get random rollout and add num-macros actions
    torch.save(agents["current"].state_dict(), LOAD_CKPT)
    macros, counts = get_macro_from_agent(NUM_MACROS, 4, USE_CUDA,
                                          AGENT, LOAD_CKPT, GRAMMAR_DIR, ENV_ID,
                                          g_type=GRAMMAR_TYPE)

    # Setup agent, replay replay_buffer, logging stats df
    if AGENT == "MLP-DQN" or AGENT == "DOUBLE":
        agents, optimizer = init_agent(MLP_DQN, L_RATE, USE_CUDA, NUM_ACTIONS)
    elif AGENT == "MLP-Dueling-DQN":
        agents, optimizer = init_agent(MLP_DDQN, L_RATE, USE_CUDA, NUM_ACTIONS)
    elif AGENT == "CNN-Dueling-DQN":
        agents, optimizer = init_agent(CNN_DDQN, L_RATE, USE_CUDA, NUM_ACTIONS)

    replay_buffer = ReplayBuffer(capacity=5000)
    macro_buffer = MacroBuffer(capacity=1000)

    reward_stats = pd.DataFrame(columns=["opt_counter", "rew_mean", "rew_sd",
                                         "rew_median", "rew_10th_p", "rew_90th_p"])

    step_stats = pd.DataFrame(columns=["opt_counter", "steps_mean", "steps_sd",
                                       "steps_median", "steps_10th_p", "steps_90th_p"])

    # Initialize optimization update counter and environment
    opt_counter = 0
    if ENV_ID == "dense-v0":
        env = gym.make(ENV_ID)
    else:
        # Wrap the ATARI env in DeepMind Wrapper
        env = make_atari(ENV_ID)
        env = wrap_deepmind(env, episode_life=True, clip_rewards=True,
                            frame_stack=True, scale=True)
        env = wrap_pytorch(env)

    ep_id = 0
    # RUN TRAINING LOOP OVER EPISODES
    while opt_counter < NUM_UPDATES:
        epsilon = epsilon_by_episode(ep_id + 1, EPS_START, EPS_STOP, EPS_DECAY)

        obs = env.reset()

        steps = 0
        while steps < MAX_STEPS:
            if ENV_ID == "dense-v0":
                action = agents["current"].act(obs.flatten(), epsilon)
            else:
                action = agents["current"].act(obs, epsilon)

            if action < 4:
                next_obs, rew, done, _  = env.step(action)
                steps += 1

                # Push transition to ER Buffer
                replay_buffer.push(ep_id, steps, obs, action,
                                   rew, next_obs, done)
            else:
                # Need to execute a macro action
                macro = macros[action - 4]
                next_obs, macro_rew, done, _ = macro_action_exec(ep_id, obs,
                                                                 steps,
                                                                 replay_buffer,
                                                                 macro, env,
                                                                 GAMMA)
                steps += len(macro)
                # Push macro transition to ER Buffer
                macro_buffer.push(ep_id, steps, obs, action,
                                  macro_rew, next_obs,
                                  done, len(macro), macro)


            if len(replay_buffer) > TRAIN_BATCH_SIZE:
                opt_counter += 1
                loss = compute_td_loss(agents, optimizer, replay_buffer,
                                       TRAIN_BATCH_SIZE, GAMMA, Variable,
                                       TRAIN_DOUBLE, ENV_ID)

            # Check for Online Transfer
            if (opt_counter+1) % GRAMMAR_EVERY == 0:
                torch.save(agents["current"].state_dict(), LOAD_CKPT)
                macros, counts = get_macro_from_agent(NUM_MACROS, NUM_ACTIONS, USE_CUDA,
                                                      AGENT, LOAD_CKPT, GRAMMAR_DIR, ENV_ID,
                                                      macros, GRAMMAR_TYPE)

            # On-Policy Rollout for Performance evaluation
            if (opt_counter+1) % ROLLOUT_EVERY == 0:
                r_stats, s_stats = get_logging_stats(opt_counter, agents,
                                                     GAMMA, NUM_ROLLOUTS,
                                                     MAX_STEPS, ENV_ID)
                reward_stats = pd.concat([reward_stats, r_stats], axis=0)
                step_stats = pd.concat([step_stats, s_stats], axis=0)

            if VERBOSE and (opt_counter+1) % PRINT_EVERY == 0:
                stop = time.time()
                print(log_template.format(opt_counter+1, stop-start,
                                          r_stats.loc[0, "rew_median"],
                                          r_stats.loc[0, "rew_mean"],
                                          s_stats.loc[0, "steps_median"],
                                          s_stats.loc[0, "steps_mean"]))
                start = time.time()

            if (opt_counter+1) % UPDATE_EVERY == 0:
                update_target(agents["current"], agents["target"])

            # Go to next episode if current one terminated or update obs
            if done: break
            else: obs = next_obs

        ep_id +=1
    # Finally save all results!
    df_to_save = pd.concat([reward_stats, step_stats], axis=1)
    df_to_save = df_to_save.loc[:,~df_to_save.columns.duplicated()]
    df_to_save = df_to_save.reset_index()
    if args.SAVE:
        torch.save(agents["current"].state_dict(), "agents/online_" + AGENT_FNAME)
        # Save the logging dataframe
        df_to_save.to_csv("results/online_" + STATS_FNAME)
    return df_to_save


if __name__ == "__main__":
    dqn_args = command_line_dqn_grid(parent=True)
    all_args = command_line_grammar_dqn(dqn_args)

    if all_args.RUN_TIMES == 1:
        print("START RUNNING {} AGENT LEARNING FOR 1 TIME".format(all_args.AGENT))
        if all_args.RUN_EXPERT_GRAMMAR:
            run_smdp_dqn_learning(all_args)
        elif all_args.RUN_ONLINE_GRAMMAR:
            run_online_dqn_smdp_learning(all_args)
        else:
            run_dqn_learning(all_args)
    else:
        mp.set_start_method('forkserver', force=True)
        if all_args.RUN_EXPERT_GRAMMAR:
            run_multiple_times(all_args, run_smdp_dqn_learning)
        elif all_args.RUN_ONLINE_GRAMMAR:
            run_multiple_times(all_args, run_online_dqn_smdp_learning)
        else:
            run_multiple_times(all_args, run_dqn_learning)
