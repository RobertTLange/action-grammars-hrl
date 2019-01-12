import numpy as np


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
            assert(len(q) == self.num_actions)
            self.table[tuple(state)] = q
        else:
            self.table[tuple(state)][action] = q

    def update_all(self, Q):
        self.table = Q


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
