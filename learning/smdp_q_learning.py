import numpy as np
from agents import QTable
from utils.general import ReplayBuffer, greedy_eval, discounted_return, macro_step


def smdp_q_learning_update(gamma, alpha, lambd, q_func, eligibility,
                           cur_state, action, next_state, reward, done, stp,
                           old_greedy_choice=None, old_action=None, old_state=None):
    if done:
        target = discounted_return(reward, gamma)
    else:
        target = discounted_return(reward, gamma) + gamma**len(reward) * np.max(q_func(next_state))

    if stp > 0:
        if old_greedy_choice == action:
            eligibility(old_state)[old_action] *= gamma*lambd
        else:
            eligibility(old_state)[old_action] = 0

    eligibility(cur_state)[action] += 1

    td_err = target - q_func(cur_state)[action]
    Q_new = q_func.table + alpha* td_err * eligibility.table

    q_func.update_all(Q_new)
    return eligibility, td_err


def smdp_q_learning(env, agent, num_episodes, max_steps,
                    gamma, alpha, lambd, epsilon,
                    log_freq, log_episodes, verbose):

    log_template = "Ep: {:>2} | Avg/Std Steps: {:.2f}/{:.2f} | Avg/Std Ret: {:.2f}/{:.2f} | Success R: {:.2f}"
    log_counter = 0
    hist = np.zeros((int(num_episodes/log_freq), 6))

    # Init Replay Buffer
    er_buffer = ReplayBuffer(num_episodes*max_steps, record_macros=True)

    for ep_id in range(num_episodes):

        state = env.reset()

        stp = 0
        tot_td = 0
        rewards = []

        eligibility = QTable(np.zeros(env.num_disks*(3, ) + (6 + len(agent.macros),)))

        old_greedy_choice = None
        old_action = None
        old_state = None

        for i in range(max_steps):
            action = agent.epsilon_greedy_action(state, epsilon)

            if action > 5:
                next_state, reward, done, _ = macro_step(action, state, agent,
                                                         env, er_buffer,
                                                         ep_id)
            else:
                next_state, reward, done, _ = env.step(action)
                er_buffer.push(ep_id, state, action, reward, next_state, done, None)

            if type(reward) != list:
                reward = [reward]
            greedy_choice = agent.greedy_action(next_state)

            # Update value function
            eligibility, tde = smdp_q_learning_update(gamma, alpha, lambd, agent.q_func,
                                                 eligibility, state, action,
                                                 next_state, reward, done, stp,
                                                 old_greedy_choice, old_action, old_state)

            # Update variables
            old_state = state
            old_action = action
            old_greedy_choice = greedy_choice
            state = next_state

            # Update counters
            stp += len(reward)
            tot_td += tde
            rewards.append(reward)

            if done: break

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
