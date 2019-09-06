import time
import numpy as np
import pandas as pd
import multiprocessing as mp

import gym
import gym_hanoi

from agents.q_agent import Agent_Q
from agents.smdp_q_agent import SMDP_Agent_Q, Macro

from utils.q_helpers_towers import q_learning
from utils.smdp_helpers_towers import smdp_q_learning, smdp_q_online_learning
from utils.general_towers import get_optimal_macros, command_line_towers, DotDic, learning_params

SEQ_DIR = "grammars/sequitur/"
LEXIS_DIR = "grammars/Lexis/"

def run_learning(args):
    LEARN_TYPE = args.LEARN_TYPE
    GRAMMAR_TYPE = args.GRAMMAR_TYPE
    TRANSFER_DISTANCE = args.TRANSFER_DISTANCE

    N_DISKS = args.N_DISKS
    NUM_ROLLOUTS = args.NUM_ROLLOUTS
    ROLLOUT_EVERY = args.ROLLOUT_EVERY
    PRINT_EVERY = args.PRINT_EVERY
    VERBOSE = args.VERBOSE

    RUN_TIMES = args.RUN_TIMES
    STATS_FNAME = args.SAVE_FNAME

    params = DotDic(learning_params(LEARN_TYPE, N_DISKS))
    ALPHA = params.ALPHA
    GAMMA = params.GAMMA
    LAMBDA = params.LAMBDA
    EPSILON = params.EPSILON
    NUM_UPDATES = params.NUM_UPDATES
    MAX_STEPS = params.MAX_STEPS

    GRAMMAR_TYPE = args.GRAMMAR_TYPE
    if GRAMMAR_TYPE == "2-Sequitur" or GRAMMAR_TYPE == "3-Sequitur":
        GRAMMAR_DIR = SEQ_DIR
        g_type = "sequitur"
    elif GRAMMAR_TYPE == "G-Lexis":
        GRAMMAR_DIR = LEXIS_DIR
        g_type = "lexis"

    env = gym.make("Hanoi-v0")
    env.set_env_parameters(N_DISKS, env_noise=0, verbose=False)

    df_across_runs = []

    # Initialize agent instances before starting to train - faster w. reset!
    if LEARN_TYPE == "Q-Learning":
        agent = Agent_Q(env)
    elif LEARN_TYPE == "Imitation-SMDP-Q-Learning":
        macros = get_optimal_macros(N_DISKS, GRAMMAR_TYPE)
        agent = SMDP_Agent_Q(env, macros)
    elif LEARN_TYPE == "Transfer-SMDP-Q-Learning":
        macros = get_optimal_macros(N_DISKS - TRANSFER_DISTANCE, GRAMMAR_TYPE)
        agent = SMDP_Agent_Q(env, macros)
    elif LEARN_TYPE == "Online-SMDP-Q-Learning":
        # Get Grammar Update Hyperparameter Schedule
        seq_k_schedule, seq_update_schedule = learning_params(LEARN_TYPE, N_DISKS,
                                                              ONLINE=True)

    print("START RUNNING {} AGENT LEARNING FOR {} TIMES".format(LEARN_TYPE,
                                                                args.RUN_TIMES))
    for t in range(RUN_TIMES):
        np.random.seed(t)
        start_t = time.time()
        # Reset values to 0 initialization without having to recompute mov_map
        if LEARN_TYPE != "Online-SMDP-Q-Learning":
            agent.reset_values()

        if LEARN_TYPE == "Q-Learning":
            df_temp = q_learning(agent, N_DISKS, NUM_UPDATES, MAX_STEPS,
                                 GAMMA, ALPHA, LAMBDA, EPSILON,
                                 ROLLOUT_EVERY, NUM_ROLLOUTS, STATS_FNAME,
                                 PRINT_EVERY, VERBOSE)

        elif LEARN_TYPE == "Imitation-SMDP-Q-Learning" or LEARN_TYPE == "Transfer-SMDP-Q-Learning":
            df_temp = smdp_q_learning(agent, N_DISKS, NUM_UPDATES, MAX_STEPS,
                                      GAMMA, ALPHA, LAMBDA, EPSILON,
                                      ROLLOUT_EVERY, NUM_ROLLOUTS, STATS_FNAME,
                                      PRINT_EVERY, VERBOSE)

        elif LEARN_TYPE == "Online-SMDP-Q-Learning":
            df_temp = smdp_q_online_learning(N_DISKS, NUM_UPDATES, MAX_STEPS,
                                             GAMMA, ALPHA, LAMBDA, EPSILON,
                                             ROLLOUT_EVERY, NUM_ROLLOUTS, STATS_FNAME,
                                             PRINT_EVERY, VERBOSE, GRAMMAR_DIR, g_type,
                                             seq_k_schedule, seq_update_schedule)

        df_across_runs.append(df_temp)
        total_t = time.time() - start_t
        print("Done training {}/{} runs after {:.2f} Secs".format(t+1,
                                                                  args.RUN_TIMES,
                                                                  total_t))


    df_concat = pd.concat(df_across_runs)
    by_row_index = df_concat.groupby(df_concat.index)
    df_means = by_row_index.mean()

    if LEARN_TYPE == "Transfer-SMDP-Q-Learning":
        df_means.to_csv("results/TOH/" + str(args.RUN_TIMES) + "_RUNS_" + str(args.N_DISKS) + "_DISKS_" + LEARN_TYPE  + "_" + str(TRANSFER_DISTANCE)+ "_"+ args.SAVE_FNAME)
    else:
        df_means.to_csv("results/TOH/" + str(args.RUN_TIMES) + "_RUNS_" + str(args.N_DISKS) + "_DISKS_" + LEARN_TYPE  + "_" + args.SAVE_FNAME)

    return df_means


if __name__ == "__main__":
    args = command_line_towers()
    run_learning(args)

    # python run_learning_towers.py --N_DISKS 5 --LEARN_TYPE Online-SMDP-Q-Learning  --RUN_TIMES 1 --GRAMMAR_TYPE G-Lexis --VERBOSE
