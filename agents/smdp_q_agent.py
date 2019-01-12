import numpy as np
import itertools
from agents.q_agent import Agent_Q


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
            assert(len(q) == self.num_actions)
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


class SMDP_Agent_Q(Agent_Q):
    def __init__(self, env, macros):
        super().__init__(env)
        self.q_func = SMDPQTable(env.get_movability_map(), macros)

        self.macros = macros
        self.num_macros = len(self.macros)
        self.current_macro = None
        for i, mac in enumerate(self.macros):
            mac.identifier = i


class Online_SMDP_Agent_Q(SMDP_Agent_Q):
    def __init__(self, env, macros):
        super().__init__(env)
        self.q_func = SMDPQTable(env.get_movability_map(), macros)

        self.macros = macros
        self.num_macros = len(self.macros)
        self.current_macro = None
        for i, mac in enumerate(self.macros):
            mac.identifier = i


class GrammarBuffer(object):
    # Object similar to Replay buffer that stores tabular values of macro
    def __init__(self, capacity):
        self.buffer = {}

    def push(self, grammar_macro, macro_values):
        state = state
        next_state = next_state

        if self.record_macros:
            self.buffer.append((ep_id, state, action, macro,
                                reward, next_state, done))
        else:
            self.buffer.append((ep_id, state, action,
                                reward, next_state, done))

    def retrieve(self, grammar_macro):
        return


class Macro():
    def __init__(self, env, action_seq):
        self.action_seq = action_seq
        self.macro_len = len(self.action_seq)
        self.env = env

        if type(self.action_seq) is str:
            self.action_seq = self.convert_string()

        self.current_time = 0
        self.active = False

    def convert_string(self):
        macro_actions = []
        for i, let in enumerate(self.action_seq):
            macro_actions.append(letter_to_action[let])
        return macro_actions

    def follow_macro(self):
        act_temp = self.action_seq[self.current_time]
        self.current_time += 1

        if self.current_time == self.macro_len:
            self.active = False
            self.current_time = 0

        if self.env.move_allowed(action_to_move[act_temp]):
            return act_temp
        else:
            self.active = False
            self.current_time = 0
            return None


letter_to_action = {"a": 0, "b": 1, "c": 2,
                    "d": 3, "e": 4, "f": 5}

action_to_move = {0: (0, 1), 1: (0, 2), 2: (1, 0),
                  3: (1, 2), 4: (2, 0), 5: (2, 1)}
