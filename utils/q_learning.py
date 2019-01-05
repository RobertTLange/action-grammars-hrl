import numpy as np
from utils.general import greedy_eval
from agents import ReplayBuffer

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


def q_learning(env, agent, num_episodes, max_steps,
               gamma, alpha, lambd, log_freq, log_episodes, verbose):

    log_template = "Ep: {:>2} | Avg/Std Steps: {:.2f}/{:.2f} | Avg/Std Ret: {:.2f}/{:.2f} | Success R: {:.2f}"
    log_counter = 0
    hist = np.zeros((int(num_episodes/log_freq), 6))

    # Init Replay Buffer
    er_buffer = ReplayBuffer(num_episodes*max_steps)

    for ep_id in range(num_episodes):

        cur_state = env.reset()

        stp = 0
        tot_td = 0
        rewards = []

        eligibility = np.zeros(agent.q_func.table.shape)

        old_greedy_choice = None
        old_action = None
        old_state = None

        for i in range(max_steps):
            action = agent.epsilon_greedy_action(cur_state)
            next_state, reward, done, _ = env.step(action)
            greedy_choice = agent.greedy_action(next_state)

            # Update value function
            eligibility, tde = q_learning_update(gamma, alpha, lambd, agent.q_func,
                                                 eligibility, cur_state, action,
                                                 next_state, reward, done, stp,
                                                 old_greedy_choice, old_action, old_state)

            # Extend replay buffer
            er_buffer.push(ep_id, old_state, action, reward, next_state, done)

            # Update variables
            old_state = cur_state
            old_action = action
            old_greedy_choice = greedy_choice
            cur_state = next_state

            # Update counters
            stp += 1
            tot_td += tde
            rewards.append(reward)

            # Go to next episode if successfully ended
            if done:
                break

        if ep_id % log_freq == 0:
            avg_steps, sd_steps, avg_ret, sd_ret, success_rate = greedy_eval(env, agent, gamma,
                                                                             max_steps, log_episodes)
            hist[log_counter,:] = np.array([ep_id, avg_steps, sd_steps,
                                            avg_ret, sd_ret, success_rate])
            log_counter += 1

            if verbose:
                print(log_template.format(ep_id + 1, avg_steps, sd_steps,
                                          avg_ret, sd_steps, success_rate))

    return hist, er_buffer
