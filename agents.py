import numpy as np
import itertools

class QTable():
    def __init__(self, table):
        self.table = table

    def __call__(self, state):
        try:
            qs = self.table[tuple(state)]
        except IndexError:
            qs = np.zeros(self.num_actions)
            print("WARNING: IndexError in Q-function. Returning zeros.")
        return qs

    def update_table(self, state, q, action=None):
        if action is None:
            assert(len(q)==self.num_actions)
            self.table[tuple(state)] = q
        else:
            self.table[tuple(state)][action] = q

    def update_all(self, Q):
        self.table = Q


class SMDPQTable():
    def __init__(self, q_table, macros):
        self.table = self.gen_valid_table(q_table, macros)

    def __call__(self, state):
        try:
            qs = self.table[tuple(state)]
        except IndexError:
            qs = np.zeros(self.num_actions)
            print("WARNING: IndexError in Q-function. Returning zeros.")
        return qs

    def update_table(self, state, q, action=None):
        if action is None:
            assert(len(q)==self.num_actions)
            self.table[tuple(state)] = q
        else:
            self.table[tuple(state)][action] = q

    def update_all(self, Q):
        self.table = Q

    def gen_valid_table(self, q_table, macros):

        dims = q_table.shape
        dims_temp = list(dims)
        dims_temp[-1] = len(macros)

        num_disks = len(dims) - 1

        temp_table = np.zeros(tuple(dims_temp))

        table = np.concatenate((q_table, temp_table), axis=num_disks)

        id_list = num_disks*[0] + num_disks*[1] + num_disks*[2]
        states = list(itertools.permutations(id_list, num_disks))
        for state in states:
            for i, macro in enumerate(macros):
                start_action = macro.action_seq[0]
                table[state][i+6] = table[state][start_action]
        return table


class Agent():
    def __init__(self, env):
        self.num_actions = env.action_space.n
        self.allowed = env.get_movability_map()

    def random_action(self, state):
        valid_actions = np.where(self.allowed[state] != -np.inf)[0]
        return np.random.choice(valid_actions)


class Agent_Q(Agent):
    def __init__(self, env):
        super().__init__(env)
        self.q_func = QTable(env.get_movability_map())

    def greedy_action(self, state):
        q_values = self.q_func(state)
        temp = np.where(q_values == np.max(q_values))[0]
        return np.random.choice(temp)

    def epsilon_greedy_action(self, state, eps):
        roll = np.random.random()
        if roll <= eps:
            return self.random_action(state)
        else:
            return self.greedy_action(state)


class SMDP_Agent_Q(Agent_Q):
    def __init__(self, env, macros):
        super().__init__(env)
        self.q_func = SMDPQTable(env.get_movability_map(), macros)

        self.macros = macros
        self.num_macros = len(self.macros)
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
        self.current_time += 1

        if self.current_time == self.macro_len:
            self.activity = False

        if env.move_allowed(action_to_move[act_temp]):
            return act_temp
        else:
            self.activity = False
            return None


letter_to_action = {"a": 0, "b": 1, "c": 2,
                    "d": 3, "e": 4, "f": 5}

action_to_move = {0: (0, 1), 1: (0, 2), 2: (1, 0),
                  3: (1, 2), 4: (2, 0), 5: (2, 1)}
