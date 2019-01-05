import numpy as np

def discounted_return(rewards, gamma):
    """
    Input: List of rewards and discount factor
    Output: Single scalar - cumulative discounted reward of episode
    """
    try:
        discounted = 0.0
        last_discount = 1.0
        for reward_set in rewards:
            gamma_mask = [gamma**t for t in range(len(reward_set))]
            # len(reward_set) works if rewards is a listoflists (from planner)
            discounted += np.dot(reward_set,
                                 gamma_mask) * last_discount * gamma
            last_discount = last_discount * gamma_mask[-1]
    except TypeError:
        # didn't work, so rewards is a list of floats - no recursion.
        gamma_mask = [gamma**t for t in range(len(rewards))]
        discounted = np.dot(rewards, gamma_mask)
    return discounted


def greedy_eval(env, agent, gamma, max_steps, log_episodes):
    rewards = []
    steps = []
    successes = 0

    for i in range(log_episodes):
        cur_state = env.reset()
        reward_temp = []
        stp_temp = 0

        for s in range(max_steps):
            action = agent.greedy_action(cur_state)
            next_state, reward, done, _ = env.step(action)
            reward_temp.append(reward)
            cur_state = next_state
            stp_temp += 1

            if done:
                rewards.append(discounted_return(reward_temp, gamma))
                steps.append(stp_temp)
                successes += 1
                break
    avg_steps = np.mean(steps) if len(steps) > 0 else max_steps
    sd_steps = np.std(steps) if len(steps) > 0 else 0
    avg_rewards = np.mean(rewards) if len(rewards) > 0 else 0
    sd_rewards = np.std(rewards) if len(rewards) > 0 else 0

    return avg_steps, sd_steps, avg_rewards, sd_rewards, successes/log_episodes
