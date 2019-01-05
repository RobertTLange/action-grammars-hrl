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

    def update_all(self, Q):
        self.table = Q


def q_learning_update(gamma, alpha, lambd, q_func, eligibility,
                      cur_state, action, next_state, reward, done, stp,
                      old_greedy_choice=None, old_action=None, old_state=None):
    if done:
        target = reward
    else:
        target = reward + gamma * np.max(q_func(next_state))

    eligibility[cur_state][action] += 1
    td_err = target - q_func(cur_state)[action]
    Q = q_func.table + alpha* td_err * eligibility

    if stp > 1:
        if old_greedy_choice == action:
            eligibility[old_state][old_action] *= gamma*lambd
        else:
            eligibility[old_state][old_action] = 0

    q_func.update_all(Q)
    return eligibility, td_err
