import numpy as np

from collections import deque

class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, ep_id, state, action, reward, next_state, done):
        state = state
        next_state = next_state

        self.buffer.append((ep_id, state, action, reward, next_state, done))

    def sample(self, batch_size):
        ep_id, state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return ep_id, np.concatenate(state), action, reward, np.concatenate(next_state), done

    def __len__(self):
        return len(self.buffer)


class Agent():
    def __init__(self, env):
        self.num_actions = env.action_space.n
        self.allowed = env.get_movability_map()

    def random_action(self, state):
        valid_actions = np.where(self.allowed[state] != -np.inf)[0]
        return np.random.choice(valid_actions)


class Agent_Q(Agent):
    def __init__(self, env, q_func):
        super().__init__(env)
        self.q_func = q_func

    def greedy_action(self, state):
        q_values = self.q_func(state)
        temp = np.where(q_values == np.max(q_values))[0]
        return np.random.choice(temp)

    def epsilon_greedy_action(self, state, eps=0.1):
        roll = np.random.random()
        if roll <= eps:
            return self.random_action(state)
        else:
            return self.greedy_action(state)


class SMDP_Agent_Q(Agent_Q):
    def __init__(self, env, q_func, macros):
        super().__init__(env, q_func)
        if not len(macros)==self.q_func.num_actions:
            print("WARNING: Number of options does not match Q-table dimensions")
        self.macros = macros
        self.num_macros    = len(self.macros)
        self.current_macro = None
        for i, mac in enumerate(self.macros):
            mac.identifier = i

    def pick_macro_greedy_epsilon(self, state, eps=0.0):
        valid_options = [i for i in np.arange(self.num_options) if self.options[i].check_validity(state)]
        all_qs        = self.q_func(state)
        valid_qs      = [all_qs[i] for i in valid_options]
        roll = np.random.random()
        if roll <= eps:
            self.current_option = np.random.choice(valid_options)
        else:
            self.current_option = valid_options[np.argmax(valid_qs)]
        return self.options[self.current_option]


class Macro():
    def __init__(self, env, action_seq):
        self.action_seq = action_seq
        self.macro_len = len(self.action_seq)
		self.env = env

        if type(self.action_seq) is str:
            self.action_seq = self.convert_string()

        self.current_time = 0
        self.activity = False

    def convert_string(self):
        macro_actions = []
        for i, let in enumerate(self.action_seq):
            macro_actions.append(letter_to_action[let])
        return macro_actions

    def follow_macro(self):
		act_temp = self.action_seq[self.current_time]
		# TODO: WRITE THIS FUNCTION
		env.move_allowed(action_to_move[act_temp])

		self.current_time += 1

		if self.current_time == self.macro_len:
			self.activity = False
        return act_temp


letter_to_action = {"a": 0, "b": 1, "c": 2,
                    "d": 3, "e": 4, "f": 5}

action_to_move = {0: (0, 1), 1: (0, 2), 2: (1, 0),
                  3: (1, 2), 4: (2, 0), 5: (2, 1)}
