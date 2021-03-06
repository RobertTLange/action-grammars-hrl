import gym
import string

import gridworld
from collections import deque
import argparse

import torch
from agents.dqn import CNN_DDQN, MLP_DQN, MLP_DDQN, init_agent
from utils.cfg_grammar import get_macros, letter_to_action, action_to_letter
from utils.atari_wrapper import make_atari, wrap_deepmind, wrap_pytorch


def command_line_grammar_dqn(dqn_parser):
    parser = argparse.ArgumentParser(parents=[dqn_parser])
    parser.add_argument('-l_ckpt', '--LOAD_CKPT', action="store",
                        default="agents/3000_mlp_agent.pt", type=str, help='Path from which to load expert')
    parser.add_argument('-n_macros', '--NUM_MACROS', action="store",
                        default=2, type=int, help='Number of used macros')
    parser.add_argument('-run_expert', '--RUN_EXPERT_GRAMMAR', action='store_true',
                        default=False)
    parser.add_argument('-grammar_upd', '--GRAMMAR_EVERY', action="store",
                        default=1000, type=int, help='#Updates after which to infer new grammar')
    parser.add_argument('-g_type', '--GRAMMAR_TYPE', action="store",
                        default="sequitur", type=str, help='Context-Free Grammar Type')
    parser.add_argument('-run_online', '--RUN_ONLINE_GRAMMAR', action='store_true',
                        default=False)
    return parser.parse_args()


class ReplayBuffer(object):
    def __init__(self, capacity, record_macros=False):
        self.buffer = deque(maxlen=capacity)

    def push(self, ep_id, step, state, action,
             reward, next_state, done):
        self.buffer.append((ep_id, step, state, action, reward, next_state, done))

    def sample(self, batch_size):
        ep_id, step, state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.stack(state), action, reward, np.stack(next_state), done

    def __len__(self):
        return len(self.buffer)


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


def rollout_macro_episode(agents, GAMMA, MAX_STEPS, ENV_ID, macros=None):
    if ENV_ID == "dense-v0":
        env = gym.make(ENV_ID)
    else:
        # Wrap the ATARI env in DeepMind Wrapper
        env = make_atari(ENV_ID)
        env = wrap_deepmind(env, episode_life=True, clip_rewards=True,
                            frame_stack=True, scale=True)
        env = wrap_pytorch(env)
    # Rollout the policy for a single episode - greedy!
    replay_buffer = ReplayBuffer(capacity=20000)
    obs = env.reset()
    episode_rew = 0
    steps = 0

    if ENV_ID == "dense-v0":
        NUM_PRIMITIVES = 4
    elif ENV_ID == "PongNoFrameskip-v4":
        NUM_PRIMITIVES = 6
    elif ENV_ID == "SeaquestNoFrameskip-v4":
        NUM_PRIMITIVES = 18
    elif ENV_ID == "MsPacmanNoFrameskip-v4":
        NUM_PRIMITIVES = 9

    while steps < MAX_STEPS:
        if ENV_ID == "dense-v0":
            action = agents["current"].act(obs.flatten(), epsilon=0.05)
        else:
            action = agents["current"].act(obs, epsilon=0.05)
        if action < NUM_PRIMITIVES:
            next_obs, reward, done, _  = env.step(action)
            steps += 1

            # Push transition to ER Buffer
            replay_buffer.push(0, steps, obs, int(action),
                               reward, next_obs, done)
        else:
            # Need to execute a macro action
            macro = macros[action - NUM_PRIMITIVES]
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
                         LOAD_CKPT, GRAMMAR_DIR, ENV_ID, macros=None,
                         g_type="sequitur", k=2):
    # Returns list of strings corresponding to inferred macro-actions
    if AGENT == "MLP-DQN":
        agents, optimizer = init_agent(MLP_DQN, 0, USE_CUDA, NUM_ACTIONS, LOAD_CKPT)
    elif AGENT == "MLP-Dueling-DQN":
        agents, optimizer = init_agent(MLP_DDQN, 0, USE_CUDA, NUM_ACTIONS, LOAD_CKPT)
    elif AGENT == "CNN-Dueling-DQN":
        agents, optimizer = init_agent(CNN_DDQN, 0, USE_CUDA, NUM_ACTIONS, LOAD_CKPT)

    steps, episode_rew, er_buffer = rollout_macro_episode(agents, 1, 5000, ENV_ID, macros)

    SENTENCE = []

    for step in range(len(er_buffer)):
        SENTENCE.append(action_to_letter(er_buffer[step][3]))

    SENTENCE = "".join(SENTENCE)

    if ENV_ID == "dense-v0":
        NUM_PRIMITIVES = 4
    elif ENV_ID == "PongNoFrameskip-v4":
        NUM_PRIMITIVES = 6
    elif ENV_ID == "SeaquestNoFrameskip-v4":
        NUM_PRIMITIVES = 18
    elif ENV_ID == "MsPacmanNoFrameskip-v4":
        NUM_PRIMITIVES = 9

    # Collect actions from rollout into string & call sequitur
    macros, counts, stats = get_macros(NUM_MACROS, SENTENCE, NUM_PRIMITIVES, GRAMMAR_DIR,
                                       k=k, g_type=g_type)
    return macros, counts, stats


if __name__ == "__main__":
    args = command_line_dqn_grid()
    USE_CUDA = torch.cuda.is_available()
    NUM_ACTIONS = 4
    AGENT = args.AGENT
    LOAD_CKPT = "agents/1000_mlp_agent.pt"
    SEQ_DIR = "../grammars/sequitur/"

    macros, counts = get_macro_from_agent("all", NUM_ACTIONS, USE_CUDA, AGENT,
                                          LOAD_CKPT, SEQ_DIR)
    print(macros, counts)
