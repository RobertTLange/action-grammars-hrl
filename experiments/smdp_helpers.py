import gym
import gridworld
from collections import deque
import argparse

import torch
from dqn import MLP_DQN, MLP_DDQN, init_agent
from dqn_helpers import command_line_dqn, ReplayBuffer
from cfg_grammar_dqn import get_macros


def command_line_grammar_dqn():
    parser = argparse.ArgumentParser()
    parser.add_argument('-roll_upd', '--ROLLOUT_EVERY', action="store",
                        default=50, type=int,
                        help='Rollout test performance after # batch updates.')
    parser.add_argument('-save_upd', '--SAVE_EVERY', action="store",
                        default=2000, type=int,
                        help='Save network and learning stats after # batch updates')
    parser.add_argument('-update_upd', '--UPDATE_EVERY', action="store",
                        default=100, type=int,
                        help='Update target network after # batch updates')
    parser.add_argument('-n_runs', '--RUN_TIMES', action="store",
                        default=1, type=int,
                        help='# Times to run agent learning')
    parser.add_argument('-n_upds', '--NUM_UPDATES', action="store",
                        default=100, type=int,
                        help='# Epochs to train for')
    parser.add_argument('-n_roll', '--NUM_ROLLOUTS', action="store",
                        default=10, type=int,
                        help='# rollouts for tracking learning progrees')
    parser.add_argument('-max_steps', '--MAX_STEPS', action="store",
                        default=1000, type=int,
                        help='Max # of steps before episode terminated')
    parser.add_argument('-v', '--VERBOSE', action="store_true", default=False,
                        help='Get training progress printed out')
    parser.add_argument('-print', '--PRINT_EVERY', action="store",
                        default=100, type=int,
                        help='#Episodes after which to print.')

    parser.add_argument('-gamma', '--GAMMA', action="store",
                        default=0.9, type=float,
                        help='Discount factor')
    parser.add_argument('-l_r', '--L_RATE', action="store", default=0.001,
                        type=float, help='Save network and learning stats after # epochs')
    parser.add_argument('-e_start', '--EPS_START', action="store", default=1,
                        type=float, help='Start Exploration Rate')
    parser.add_argument('-e_stop', '--EPS_STOP', action="store", default=0.01,
                        type=float, help='Start Exploration Rate')
    parser.add_argument('-e_decay', '--EPS_DECAY', action="store", default=500,
                        type=float, help='Start Exploration Rate')


    parser.add_argument('-train_batch', '--TRAIN_BATCH_SIZE', action="store",
                        default=32, type=int, help='# images in training batch')
    parser.add_argument('-agent', '--AGENT', action="store",
                        default="MLP-DQN", type=str, help='Agent model')
    parser.add_argument('-l_ckpt', '--LOAD_CKPT', action="store",
                        default="agents/3000_mlp_agent.pt", type=str, help='Path from which to load expert')
    parser.add_argument('-n_macros', '--NUM_MACROS', action="store",
                        default=2, type=int, help='Number of used macros')


    parser.add_argument('-device', '--device_id', action="store",
                        default=0, type=int, help='Device id on which to train')
    parser.add_argument('-agent_file', '--AGENT_FNAME', action="store",
                        default="expert_grammar_mlp_agent.pt", type=str,
                        help='Path to store online agents params')
    parser.add_argument('-stats_file', '--STATS_FNAME', action="store",
                        default="expert_grammar_stats.csv", type=str,
                        help='Path to store stats of MLP agent')
    return parser.parse_args()


def command_line_online_grammar_dqn():
    parser = argparse.ArgumentParser()
    parser.add_argument('-roll_upd', '--ROLLOUT_EVERY', action="store",
                        default=20, type=int,
                        help='Rollout test performance after # batch updates.')
    parser.add_argument('-save_upd', '--SAVE_EVERY', action="store",
                        default=2000, type=int,
                        help='Save network and learning stats after # batch updates')
    parser.add_argument('-update_upd', '--UPDATE_EVERY', action="store",
                        default=100, type=int,
                        help='Update target network after # batch updates')
    parser.add_argument('-n_runs', '--RUN_TIMES', action="store",
                        default=1, type=int,
                        help='# Times to run agent learning')
    parser.add_argument('-n_upds', '--NUM_UPDATES', action="store",
                        default=100, type=int,
                        help='# Epochs to train for')
    parser.add_argument('-n_roll', '--NUM_ROLLOUTS', action="store",
                        default=5, type=int,
                        help='# rollouts for tracking learning progrees')
    parser.add_argument('-max_steps', '--MAX_STEPS', action="store",
                        default=200, type=int,
                        help='Max # of steps before episode terminated')
    parser.add_argument('-v', '--VERBOSE', action="store_true", default=False,
                        help='Get training progress printed out')
    parser.add_argument('-print', '--PRINT_EVERY', action="store",
                        default=500, type=int,
                        help='#Episodes after which to print.')

    parser.add_argument('-gamma', '--GAMMA', action="store",
                        default=0.9, type=float,
                        help='Discount factor')
    parser.add_argument('-l_r', '--L_RATE', action="store", default=0.001,
                        type=float, help='Save network and learning stats after # epochs')
    parser.add_argument('-e_start', '--EPS_START', action="store", default=1,
                        type=float, help='Start Exploration Rate')
    parser.add_argument('-e_stop', '--EPS_STOP', action="store", default=0.01,
                        type=float, help='Start Exploration Rate')
    parser.add_argument('-e_decay', '--EPS_DECAY', action="store", default=100,
                        type=float, help='Start Exploration Rate')


    parser.add_argument('-train_batch', '--TRAIN_BATCH_SIZE', action="store",
                        default=32, type=int, help='# images in training batch')
    parser.add_argument('-agent', '--AGENT', action="store",
                        default="MLP-DQN", type=str, help='Agent model')
    parser.add_argument('-l_ckpt', '--LOAD_CKPT', action="store",
                        default="agents/online_mlp_agent.pt", type=str, help='Path from which to load expert')
    parser.add_argument('-n_macros', '--NUM_MACROS', action="store",
                        default=2, type=int, help='Number of used macros')
    parser.add_argument('-grammar_upd', '--GRAMMAR_EVERY', action="store",
                        default=2, type=int, help='#Updates after which to infer new grammar')


    parser.add_argument('-device', '--device_id', action="store",
                        default=0, type=int, help='Device id on which to train')
    parser.add_argument('-agent_file', '--AGENT_FNAME', action="store",
                        default="online_grammar_mlp_agent.pt", type=str,
                        help='Path to store online agents params')
    parser.add_argument('-stats_file', '--STATS_FNAME', action="store",
                        default="expert_grammar_stats.csv", type=str,
                        help='Path to store stats of MLP agent')
    return parser.parse_args()


class MacroBuffer(object):
    def __init__(self, capacity, record_macros=False):
        self.buffer = deque(maxlen=capacity)

    def push(self, ep_id, step, state, action,
             reward, next_state, done, tau, string_action):
        self.buffer.append((ep_id, step, state, action,
                            reward, next_state, done, tau, string_action))

    def sample(self, batch_size):
        if not self.record_macros:
            ep_id, step, state, action, reward, next_state, done, tau, active = zip(*random.sample(self.buffer, batch_size))
            return np.stack(state), action, reward, np.stack(next_state), done

    def __len__(self):
        return len(self.buffer)


def macro_action_exec(ep_id, obs, steps, replay_buffer, macro, env, GAMMA):
    # Macro is a sequence of strings corresponding to primitive actions
    macro_rew = 0
    for i, string_action in enumerate(macro):
        # Decode string to primitive action
        action = letter_to_action(string_action)
        next_obs, rew, done, _  = env.step(action)
        # Push primitive transition to ER Buffer
        replay_buffer.push(ep_id, steps+i+1, obs, action,
                           rew, next_obs, done)
        # Accumulate macro reward
        obs = next_obs
        macro_rew += GAMMA**i * rew

        if done:
            break
    return next_obs, macro_rew, done, _


def letter_to_action(string_action):
    dic = {"a": 0, "b": 1, "c": 2, "d": 3}
    return dic[string_action]


def action_to_letter(action):
    dic = {0: "a", 1: "b", 2: "c", 3: "d"}
    return dic[action]


def rollout_macro_episode(agents, GAMMA, MAX_STEPS):
    env = gym.make("dense-v0")
    # Rollout the policy for a single episode - greedy!
    replay_buffer = ReplayBuffer(capacity=5000)
    obs = env.reset()
    episode_rew = 0
    steps = 0

    while steps < MAX_STEPS:
        action = agents["current"].act(obs.flatten(), epsilon=0.05)
        if action < 4:
            next_obs, reward, done, _  = env.step(action)
            steps += 1

            # Push transition to ER Buffer
            replay_buffer.push(0, steps, obs, int(action),
                               reward, next_obs, done)
        else:
            # Need to execute a macro action
            macro = macros[action - 4]
            next_obs, reward, done, _ = macro_action_exec(0, obs, steps,
                                                          replay_buffer,
                                                          macro, env,
                                                          GAMMA)
            steps += len(macro)

        episode_rew += GAMMA**(steps - 1) * reward
        if done:
            break
    return steps, episode_rew, replay_buffer.buffer


def get_macro_from_agent(NUM_MACROS, NUM_ACTIONS, USE_CUDA, AGENT,
                         LOAD_CKPT, SEQ_DIR):
    # Returns list of strings corresponding to inferred macro-actions
    macros = []
    if AGENT == "MLP-DQN":
        agents, optimizer = init_agent(MLP_DQN, 0, USE_CUDA, NUM_ACTIONS, LOAD_CKPT)
    elif AGENT == "MLP-Dueling-DQN":
        agents, optimizer = init_agent(MLP_DDQN, 0, USE_CUDA, NUM_ACTIONS, LOAD_CKPT)

    steps, episode_rew, er_buffer = rollout_macro_episode(agents, 1, 200)

    SENTENCE = []

    for step in range(len(er_buffer)):
        SENTENCE.append(action_to_letter(er_buffer[step][3]))

    SENTENCE = "".join(SENTENCE)
    # Collect actions from rollout into string & call sequitur
    macros, counts = get_macros(NUM_MACROS, SENTENCE, 4, SEQ_DIR, k=2)
    return macros, counts


if __name__ == "__main__":
    args = command_line_dqn()
    USE_CUDA = torch.cuda.is_available()
    NUM_ACTIONS = 4
    AGENT = args.AGENT
    LOAD_CKPT = "agents/1000_mlp_agent.pt"
    SEQ_DIR = "../grammars/sequitur/"

    macros, counts = get_macro_from_agent("all", NUM_ACTIONS, USE_CUDA, AGENT,
                                          LOAD_CKPT, SEQ_DIR)
    print(macros, counts)
