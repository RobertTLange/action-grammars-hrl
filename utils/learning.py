import numpy as np

class QTable():
    def __init__(self, table):
        self.table = table

    def __call__(self,state):
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


def q_learning_update(gamma, alpha, qfunc, cur_state, action, next_state, reward):
    target = reward + gamma * np.max(qfunc(next_state))
    td_err = target - qfunc(cur_state)[action]
    qfunc.update_table(cur_state,qfunc(cur_state)[action] + alpha * td_err,action)
    return td_err
