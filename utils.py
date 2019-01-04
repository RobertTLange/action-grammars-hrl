import string
import random
import time
import numpy as np
from scfg_grammar import *

def softmax_action(q_table, state):
    """
    Input: Q table and the current state of the agent
    Output: Softmax exploration sampled action
    """
    q_values = q_table[state, :]

    def softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    p_a_s = softmax(q_values)

    sampled_action = np.random.choice(q_values.shape[0], 1, p=p_a_s)
    return int(sampled_action)


def random_action(num_actions):
    """
    Output: uniformly sampled action
    """
    return np.random.randint(low=0, high=num_actions)


def greedy_action(q_table, state):
    """
    Output: greedy/best action from current state given q table estimate
    """
    q_values = q_table[state, :]
    actions = np.argwhere(q_values == np.amax(q_values))
    if len(actions) > 1:
        idx = random.sample(xrange(len(actions)), 1)[0]
        action = actions[idx][0]
    else:
        action = actions[0][0]
    return action


def epsilon_greedy_action(q_table, state, eps=0.1):
    """
    Input: Q table, current state of the agent, eps exploration parameter
    Output: Eps-greedy sampled action
    """
    num_actions = q_table.shape[1]

    roll = np.random.random()
    if roll <= eps:
        return random_action(num_actions)
    else:
        return greedy_action(q_table, state)


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


def rollout(env, Q, gamma, num_rollouts, max_steps,
            macros=None, macro_dictionary=None,
            options=None, option_params=None, successor_memory=None,
            return_traces=False, render=False, hmm=False):
    num_primitives =  env.action_space.n
    reward_list = []
    steps_list = []
    action_list = []
    state_list = []

    unique_states = set([])
    no_obs = env.observation_space.n

    for i in range(num_rollouts):
        state = env.reset()

    	steps = 0
    	rewards = []

    	action_episode = []
    	state_episode = [state]
        unique_states.add(state)
    	while steps < max_steps:
            if render:
            	env.render()
            steps += 1
            action = greedy_action(Q, state)
            if action >= num_primitives:

                if macros is not None:
                    macro_action = macros[action - num_primitives]
                    new_states, rews, done, primitives = macro_step(env, macro_action, state, macro_dictionary)
                elif options is not None:
                    option = options[action - num_primitives]
                    new_states, rews, done, primitives = option_step(env, state, option,
                                                                     option_params[0],
                                                                     option_params[1],
                                                                     option_params[2][0],
                                                                     successor_memory)
                    new_states = [state[0] for state in new_states]

                state = new_states[-1]
                rewards.extend(rews)
                action_episode.extend(primitives)
                state_episode.extend(new_states)

                for state in new_states:
                    unique_states.add(state)
            else:
            	new_state, reward, done, _ = env.step(action)
            	state = new_state
            	rewards.append(reward)
            	action_episode.append(action)
            	state_episode.append(state)
                unique_states.add(state)
            if done:
            	break

    	reward_list.append(discounted_return(rewards, gamma))
    	steps_list.append(steps)
    	action_list.append(action_episode)
    	state_list.append(state_episode)

    if hmm:
        unobs_states = set(range(no_obs)).difference(unique_states)
        for state in list(unobs_states):
            state_list.append([state])
    if return_traces:
    	return reward_list, steps_list, action_list, state_list
    else:
    	return reward_list, steps_list


def gen_macro_dictionary(num_primitives):
    action_to_string = list(string.ascii_lowercase)[:num_primitives]
    macro_dictionary = dict(zip(action_to_string, range(num_primitives)))
    return macro_dictionary


def macro_step(env, macro_action, state, macro_dictionary):
    new_states = []
    rewards = []
    primitives = []

    for i in range(len(macro_action)):
        primitive = macro_dictionary[macro_action[i]]
        new_state, reward, done, _ = env.step(primitive)
        new_states.append(new_state)
        rewards.append(reward)
        primitives.append(primitive)
        if done:
            break

    return new_states, rewards, done, primitives


def encode_actions(action_list, num_primitives):
    action_to_string = list(string.ascii_lowercase)[:num_primitives]
    encode_dictionary = dict(zip(range(num_primitives), action_to_string))

    num_seqs = len(action_list)
    encoded_seqs = []

    for i in range(num_seqs):
        temp = []
        action_seq = action_list[i]
        for j, number in enumerate(action_seq):
            letter = encode_dictionary[number]
            temp.append(letter)
        encoded_seqs.append("".join(temp))

    return encoded_seqs


def update_log(learning_log, learning_log_new, warm_runs, update_counter, eps_between_updates):
    log_eps = [learn_log[0] + warm_runs + update_counter*eps_between_updates for learn_log in learning_log_new]
    for i in range(len(learning_log_new)):
        learn_temp = learning_log_new[i]
        learn_temp[0] = log_eps[i]
        learning_log.append(learn_temp)

    return learning_log
