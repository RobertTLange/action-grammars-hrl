import gym
import math
import numpy as np
from grammars.cfg_grammar import *
from grammars.scfg_grammar import *
from utils.utils import *

def q_learning(GAME, gamma, learning_rate, epsilon, lambd, max_steps,
			   log_episode, num_episodes, num_rollouts=10, num_return_traces=0, print_out=False):
	# Implements basic Q(lambda)-Learning for discrete space OpenAI environments
	env = gym.make(GAME)

	Q = np.zeros([env.observation_space.n, env.action_space.n])

	learning_log = []

	for i in range(num_episodes):
		state = env.reset()
		done = False
		steps = 0

		eligibility = np.zeros(Q.shape)

		if log_episode != 0:
			if (i%log_episode) == 0:
				reward_list, steps_list = rollout(env, Q, gamma, num_rollouts, max_steps)
				avg_reward, std_reward = np.average(reward_list), np.std(reward_list)
				avg_steps, std_steps = np.average(steps_list), np.std(steps_list)
				learning_log.append([i, avg_reward, std_reward, avg_steps, std_steps])
				if print_out:
					print("Episode {} - Avg Rewards: {}; Std Rewards: {}".format(i, avg_reward, std_reward))
					print("Episode {} - Avg Steps: {}; Std Steps: {}".format(i, avg_steps, std_steps))

		while steps < max_steps and not done:
			steps += 1
			action = epsilon_greedy_action(Q, state, epsilon)
			new_state, reward, done, _ = env.step(action)
			greedy_choice = greedy_action(Q, new_state)

			if done:
				target = reward
			else:
				target = reward + gamma * np.max(Q[new_state,:])

			eligibility[state, action] += 1
			td_err = target - Q[state, action]
			Q += learning_rate* td_err * eligibility

			if steps > 1:
				if greedy_ch_old == action:
					eligibility[old_state, old_action] *= gamma*lambd
				else:
					eligibility[old_state, old_action] = 0

			greedy_ch_old = greedy_choice
			old_state = state
			old_action = action
			state = new_state

	if num_return_traces > 0:
		reward_list, steps_list, action_list, state_list = rollout(env, Q, gamma, num_return_traces,
																   max_steps, return_traces=True)
		return Q, learning_log, action_list, state_list
	else:
		return Q, learning_log


def smdp_q(GAME, macros, gamma, learning_rate, epsilon, lambd, max_steps,
		   log_episode, num_episodes, num_rollouts=10, num_return_traces=0, print_out=False):
	# Implements basic SMDP-Q(lambda)-Learning for discrete space OpenAI environments
	# Need to predefine a list of macros which consist of alphabetic letters corresponding to primitive actions
	env = gym.make(GAME)

	num_primitives =  env.action_space.n
	num_macros = len(macros)
	Q = np.zeros([env.observation_space.n, num_primitives + num_macros])

	macro_dictionary = gen_macro_dictionary(num_primitives)
	learning_log = []

	for i in range(num_episodes):
		state = env.reset()
		done = False
		steps = 0

		eligibility = np.zeros(Q.shape)

		if log_episode != 0:
			if (i%log_episode) == 0:
				reward_list, steps_list = rollout(env, Q, gamma, num_rollouts, max_steps, macros, macro_dictionary)
				avg_reward, std_reward = np.average(reward_list), np.std(reward_list)
				avg_steps, std_steps = np.average(steps_list), np.std(steps_list)
				learning_log.append([i, avg_reward, std_reward, avg_steps, std_steps])
				if print_out:
					print("Episode {} - Avg Rewards: {}; Std Rewards: {}".format(i, avg_reward, std_reward))
					print("Episode {} - Avg Steps: {}; Std Steps: {}".format(i, avg_steps, std_steps))

		while steps < max_steps and not done:
			steps += 1
			action = epsilon_greedy_action(Q, state, epsilon)

			if action >= num_primitives:
				macro_action = macros[action - num_primitives]
				new_states, rewards, done, _ = macro_step(env, macro_action, state, macro_dictionary)
				greedy_choice = greedy_action(Q, new_states[-1])

				if done:
					target = discounted_return(rewards, gamma)
				else:
					target = discounted_return(rewards, gamma) + gamma**len(rewards) * np.max(Q[new_states[-1],:])

			else:
				new_state, reward, done, _ = env.step(action)
				greedy_choice = greedy_action(Q, new_state)

				if done:
					target = reward
				else:
					target = reward + gamma * np.max(Q[new_state,:])

			eligibility[state, action] += 1
			td_err = target - Q[state, action]
			Q += learning_rate* td_err * eligibility

			if steps > 1:
				if greedy_ch_old == action:
					eligibility[old_state, old_action] *= gamma*lambd
				else:
					eligibility[old_state, old_action] = 0

			greedy_ch_old = greedy_choice
			old_state = state
			old_action = action

			if action >= num_primitives:
				state = new_states[-1]
			else:
				state = new_state

	if num_return_traces > 0:
		reward_list, steps_list, action_list, state_list = rollout(env, Q, gamma, num_return_traces,
																   max_steps, macros, macro_dictionary, return_traces=True)
		return Q, learning_log, action_list, state_list
	else:
		return Q, learning_log




def smdp_q_transfer(GAME, Q_old, macros, macros_old, gamma, learning_rate, epsilon, lambd, max_steps,
                    log_episode, num_episodes, num_rollouts, num_return_traces=0, print_out=False):
	# Implements SMDP-Q(lambda)-Learning for discrete space OpenAI environments
	# With transfer of old values from Q_old - check if macros were already in action space beforehand
	env = gym.make(GAME)

	num_states = env.observation_space.n
	num_primitives =  env.action_space.n

	num_macros = len(macros)
	Q = np.zeros([num_states, num_primitives + num_macros])
	Q[:, :num_primitives] = Q_old[:, :num_primitives]
	for i, m in enumerate(macros):
		if m in macros_old:
			idx = macros_old.index(m)
			Q[:, num_primitives + i] = Q_old[:, num_primitives + idx]
	macro_dictionary = gen_macro_dictionary(num_primitives)
	learning_log = []

	for i in range(num_episodes):
		state = env.reset()
		done = False
		steps = 0

		eligibility = np.zeros(Q.shape)

		if log_episode != 0:
			if (i%log_episode) == 0:
				reward_list, steps_list = rollout(env, Q, gamma, num_rollouts, max_steps, macros, macro_dictionary)
				avg_reward, std_reward = np.average(reward_list), np.std(reward_list)
				avg_steps, std_steps = np.average(steps_list), np.std(steps_list)
				learning_log.append([i, avg_reward, std_reward, avg_steps, std_steps])
				if print_out:
					print("Episode {} - Avg Rewards: {}; Std Rewards: {}".format(i, avg_reward, std_reward))
					print("Episode {} - Avg Steps: {}; Std Steps: {}".format(i, avg_steps, std_steps))

		while steps < max_steps and not done:
			steps += 1
			action = epsilon_greedy_action(Q, state, epsilon)

			if action >= num_primitives:
				macro_action = macros[action - num_primitives]
				new_states, rewards, done, _ = macro_step(env, macro_action, state, macro_dictionary)
				greedy_choice = greedy_action(Q, new_states[-1])

				if done:
					target = discounted_return(rewards, gamma)
				else:
					target = discounted_return(rewards, gamma) + gamma**len(rewards) * np.max(Q[new_states[-1],:])

			else:
				new_state, reward, done, _ = env.step(action)
				greedy_choice = greedy_action(Q, new_state)

				if done:
					target = reward
				else:
					target = reward + gamma * np.max(Q[new_state,:])

			eligibility[state, action] += 1
			td_err = target - Q[state, action]
			Q += learning_rate* td_err * eligibility

			if steps > 1:
				if greedy_ch_old == action:
					eligibility[old_state, old_action] *= gamma*lambd
				else:
					eligibility[old_state, old_action] = 0

			greedy_ch_old = greedy_choice
			old_state = state
			old_action = action

			if action >= num_primitives:
				state = new_states[-1]
			else:
				state = new_state

	if num_return_traces > 0:
		reward_list, steps_list, action_list, state_list = rollout(env, Q, gamma, num_return_traces,
																   max_steps, macros, macro_dictionary, return_traces=True)
		return Q, learning_log, action_list, state_list
	else:
		return Q, learning_log


def imitation_macro_learning(GAME, gamma, learning_rate, epsilon, lambd, max_steps,
							num_g_train_traces, warm_runs, g_type, k,
							log_episode, num_episodes, num_rollouts=10, print_out=False):
	# Implements imitation SMDP-Q(lambda)-Learning for discrete space OpenAI environments
	# Trains Q-Learner for warm_runs episodes and afterwards extracts macros using specific grammar
	Q, learning_log, action_list, state_list = q_learning(GAME, gamma, learning_rate, epsilon, 0, max_steps,
			   				    	 		               0, num_episodes, num_return_traces=num_g_train_traces)

	env = gym.make(GAME)
	macros = get_macros_from_traces(env, "all", action_list, g_type, k)

	Q, learning_log = smdp_q(GAME, macros, gamma, learning_rate, epsilon, lambd, max_steps,
			   				 log_episode, num_episodes, num_rollouts, 0, print_out)

	return Q, learning_log


def online_macro_learning(GAME, gamma, learning_rate, epsilon, lambd, max_steps,
						  num_g_train_traces, warm_runs, g_type, k, num_updates, transfer,
						  log_episode, num_episodes, num_rollouts=10, print_out=False):
	# Implements online SMDP-Q(lambda)-Learning for discrete space OpenAI environments
	# Trains Q-Learner for warm_runs episodes and afterwards alters between extracting macros using specific grammar and updating value estimates
    Q, learning_log, action_list, state_list = q_learning(GAME, gamma, learning_rate, epsilon, 0, max_steps, log_episode, warm_runs, num_return_traces=num_g_train_traces)
    env = gym.make(GAME)
    macros = get_macros_from_traces(env, "all", action_list, g_type, k)

    macro_log = []
    macro_log.append([learning_log[-1][0]+5, len(macros), sum(map(len, macros))/len(macros)])
    if print_out:
        print("Done with warm up. Got {} macros".format(len(macros)))
        print("Current performance: {} Episode: {} Avg Steps, {} Std Steps".format(learning_log[-1][0]+5, learning_log[-1][3], learning_log[-1][4]))
        print(macros)

    update_counter = 0
    eps_between_updates = int(math.ceil((num_episodes - warm_runs)/num_updates))

    while update_counter < num_updates:
        if transfer == False or update_counter == 0:
        	Q, learning_log_new, action_list, state_list = smdp_q(GAME, macros, gamma, learning_rate, epsilon, lambd, max_steps,
        							                              log_episode, eps_between_updates, num_rollouts, num_g_train_traces, False)
        else:
            Q, learning_log_new, action_list, state_list = smdp_q_transfer(GAME, Q, macros, macros_old, gamma, learning_rate, epsilon, lambd, max_steps,
        							                                       log_episode, eps_between_updates, num_rollouts, num_g_train_traces, False)

        learning_log = update_log(learning_log, learning_log_new, warm_runs, update_counter, eps_between_updates)
        update_counter += 1

        macros_old = macros
        macros = get_macros_from_traces(env, "all", action_list, g_type)
        macro_log.append([learning_log[-1][0]+5, len(macros), sum(map(len, macros))/len(macros)])
        if print_out:
            print("Done with Update {}. Got {} macros".format(update_counter, len(macros)))
            print("Current performance: {} Episode: {} Avg Steps, {} Std Steps".format(learning_log[-1][0]+5, learning_log[-1][3], learning_log[-1][4]))

    return Q, learning_log, macro_log

def option_smdp_q(GAME, Q_old, options, option_params, successor_memory,
                  gamma, learning_rate, epsilon, max_steps,
                  log_episode, num_episodes, num_rollouts, num_return_traces=0, print_out=False):
	# Implements SMDP-Q-Learning for options
    env = gym.make(GAME)

    num_states = env.observation_space.n
    num_primitives = env.action_space.n
    num_options = len(options)
    Q = np.zeros([num_states, num_primitives + num_options])

    if Q_old is not None:
        Q[:, :num_primitives] = Q_old[:, :num_primitives]

    learning_log = []

    for i in range(num_episodes):
        start = time.time()
        state = env.reset()
        done = False
        steps = 0
        if log_episode != 0:
            if (i%log_episode) == 0:
                reward_list, steps_list = rollout(env, Q, gamma, num_rollouts, max_steps,
                                                  options=options, option_params=option_params, successor_memory=successor_memory)
                avg_reward, std_reward = np.average(reward_list), np.std(reward_list)
                avg_steps, std_steps = np.average(steps_list), np.std(steps_list)
                learning_log.append([i, avg_reward, std_reward, avg_steps, std_steps])
                if print_out:
                    print("Episode {} - Avg Rewards: {}; Std Rewards: {}".format(i, avg_reward, std_reward))
                    print("Episode {} - Avg Steps: {}; Std Steps: {}".format(i, avg_steps, std_steps))

        while steps < max_steps and not done:
            steps += 1
            action = epsilon_greedy_action(Q, state, epsilon)
            if action >= num_primitives:
                selected_option = options[action - num_primitives]
                try:
                    state = state[0]
                except:
                    continue
                new_states, rewards, done, actions = option_step(env, state, selected_option,
                                                                 option_params[0], option_params[1], option_params[2][0],
                                                                 successor_memory=successor_memory)
                if done:
                    target = discounted_return(rewards, gamma)
                else:
                    target = discounted_return(rewards, gamma) + gamma**len(rewards) * np.max(Q[new_states[-1],:])
                new_state = new_states[-1]
            else:
                new_state, reward, done, _ = env.step(action)
                if done:
                    target = reward
                else:
                    target = reward + gamma * np.max(Q[new_state,:])
            td_err = target - Q[state, action]
            Q[state, action] += learning_rate* td_err
            state = new_state

    if num_return_traces > 0:
        reward_list, steps_list, action_list, state_list = rollout(env, Q, gamma, num_rollouts, max_steps,
                                                                   options=options, option_params=option_params,
                                                                   successor_memory=successor_memory, return_traces=True,
																   hmm=(option_params[2][0]=="hmm"))

        return Q, learning_log, action_list, state_list
    else:
        return Q, learning_log


def imitiation_options_learning(GAME, gamma, learning_rate, epsilon, max_steps,
						    	num_g_train_traces, warm_runs, option_params,
						        log_episode, num_episodes, num_rollouts=10, print_out=False):

    Q, learning_log = q_learning(GAME, gamma, learning_rate, epsilon, 0, max_steps, 0, num_episodes)
    env = gym.make(GAME)
    reward_list, steps_list, action_list, state_list = rollout(env, Q, gamma, num_g_train_traces, max_steps, return_traces=True, hmm=(option_params[2][0]=="hmm"))

    options = [train_grammars(env, state_list, option_params[2])]

    successor_memory = successor_buffer(env.action_space.n)

    Q, learning_log = option_smdp_q(GAME, None, options, option_params, successor_memory,
                                    gamma, learning_rate, epsilon, max_steps,
                                    log_episode, num_episodes, num_rollouts, num_return_traces=0, print_out=print_out)

    return Q, learning_log


def online_options_learning(GAME, gamma, learning_rate, epsilon, max_steps,
						    num_g_train_traces, warm_runs, num_options, option_params, num_updates,
						    log_episode, num_episodes, num_rollouts=10, print_out=False):

	Q, learning_log = q_learning(GAME, gamma, learning_rate, epsilon, 0, max_steps, log_episode, warm_runs)
	env = gym.make(GAME)
	reward_list, steps_list, action_list, state_list = rollout(env, Q, gamma, num_g_train_traces, max_steps, return_traces=True, hmm=(option_params[2][0]=="hmm"))

	options = [train_grammars(env, state_list, option_params[2])]

	successor_memory = successor_buffer(env.action_space.n)

	if print_out:
		print("Done with warm up. Trained {} options.".format(num_options))
		print("Current performance: {} Episode: {} Avg Steps, {} Std Steps".format(learning_log[-1][0]+log_episode, learning_log[-1][3], learning_log[-1][4]))

	update_counter = 0
	eps_between_updates = int(math.ceil((num_episodes - warm_runs)/num_updates))

	while update_counter < num_updates:
		Q, learning_log_new, action_list, state_list = option_smdp_q(GAME, Q, options, option_params, successor_memory,
                                                                     gamma, learning_rate, epsilon, max_steps,
                                                                     log_episode, eps_between_updates, num_rollouts,
                                                                     num_return_traces=num_g_train_traces, print_out=print_out)
		learning_log = update_log(learning_log, learning_log_new,
								  warm_runs, update_counter,
								  eps_between_updates)
		update_counter += 1
		options = [train_grammars(env, state_list, option_params[2])]

		if print_out:
			print("Done with Update {}. ".format(update_counter))
			print("Current performance: {} Episode: {} Avg Steps, {} Std Steps".format(learning_log[-1][0]+log_episode, learning_log[-1][3], learning_log[-1][4]))

	return Q, learning_log


if __name__ == '__main__':
	num_episodes = 2000
	max_steps = 100
	learning_rate = 0.8
	gamma = 0.95
	epsilon = 0.1
	lambd = 0

	num_updates = 10
	num_g_train_traces = 4
	warm_runs = 50
	g_type = "sequitur"
	k = 4
	transfer = False

	log_episode = 5
	GAME = "Taxi-v2"
	test_options = True
	if test_options:
		print("strt")
		num_options = 1
		num_updates = 10

	    # max_steps, max_surprisal, model_type
		option_params = [2, 10, ["hmm", 2]]
		# option_params = [2, 10, ["rnn", "lstm", 1]]
		imitiation_options_learning(GAME, gamma, learning_rate, epsilon, max_steps,
									num_g_train_traces, warm_runs, option_params,
									5, 2000, num_rollouts=10, print_out=True)
		online_options_learning(GAME, gamma, learning_rate, epsilon, max_steps,
							    num_g_train_traces, warm_runs, num_options, option_params, num_updates,
							    log_episode, num_episodes, num_rollouts=10, print_out=True)


    # test = False
    # if test:
    #     Q, learning_log, action_list, state_list = q_learning(GAME, gamma, learning_rate, epsilon, 0, max_steps,
    #     		   					 	 		  100, num_episodes, num_return_traces=10, print_out=True)
	#
    #     Q, learning_log, action_list, state_list = q_learning(GAME, gamma, learning_rate, epsilon, 0.1, max_steps,
    #     		   					 	 		  100, num_episodes, num_return_traces=10, print_out=True)
	#
    #     macros = ["bb", "cd", "abd"]
    #     Q, learning_log = smdp_q(GAME, macros, gamma, learning_rate, epsilon, 0,
    #     						 max_steps, 100, num_episodes, print_out=True)
	#
	#
    #     Q, learning_log = imitation_macro_learning(GAME, gamma, learning_rate, epsilon, 0, max_steps,
    #     										  num_g_train_traces, warm_runs, g_type, k,
    #     										  100, num_episodes, num_rollouts=10, print_out=True)
	#
    #     Q, learning_log = imitation_macro_learning(GAME, gamma, learning_rate, epsilon, 0, max_steps,
    #     										  num_g_train_traces, warm_runs, g_type, k,
    #     										  100, num_episodes, num_rollouts=10, print_out=True)
	#
    #     Q, learning_log, macro_log = online_macro_learning(GAME, gamma, learning_rate, epsilon, 0, max_steps,
    #     					  num_g_train_traces, warm_runs, g_type, k, num_updates, False,
    #     					  log_episode, num_episodes, num_rollouts=10, print_out=True)
	#
    #     Q, learning_log, macro_log = online_macro_learning(GAME, gamma, learning_rate, epsilon, 0.1, max_steps,
    #     					  num_g_train_traces, warm_runs, g_type, k, num_updates, True,
    #     					  log_episode, num_episodes, num_rollouts=10, print_out=True)
