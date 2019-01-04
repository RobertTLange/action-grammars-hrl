from __future__ import print_function

import numpy as np
import os
import time
import argparse

from agents import *
from utils import *
from results_proc import *


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-env', '--environment', default="Taxi-v2",
						help="RL environment")
	parser.add_argument('-s', '--show', action="store_true",
						help="Show finished, trained attempt")
	parser.add_argument('-e', '--episodes', action="store", default=2000, type=int,
						help='Number of episodes used to train')
	parser.add_argument('-ms', '--max_steps', action="store", default=100, type=int,
						help='Maximum number of steps in an episode')
	parser.add_argument('-lr', '--learning_rate', action="store", default=0.8, type=float,
						help='Learning rate for Q learning')
	parser.add_argument('-g', '--gamma', action="store", default=0.95, type=float,
						help='Discount factor')
	parser.add_argument('-eps', '--epsilon', action="store", default=0.1, type=float,
						help='Discount factor')
	parser.add_argument('-l', '--lambd', action="store", default=0.1, type=float,
						help='Eligibility Lambda')
	parser.add_argument('-l_type', '--learning_type', default="q_learning")

	args = parser.parse_args()

	GAME = args.environment
	num_episodes = args.episodes
	max_steps = args.max_steps
	learning_rate = args.learning_rate
	gamma = args.gamma
	lambd = args.lambd
	epsilon = args.epsilon

	log_episode = 5
	learning_times = 5

	base_dir = os.getcwd()
	results_dir = base_dir + "/results_options/"

	print("Running {} Learning Algorithms for {}".format(args.learning_type, GAME))

	if args.learning_type == "q_learning":
		results = average_results(func=q_learning,
								  inputs=[GAME, gamma, learning_rate, epsilon, 0, max_steps, log_episode],
								  num_episodes=num_episodes, learning_times=learning_times, save=True,
								  title=results_dir + GAME + "_q.txt")

		results = average_results(func=q_learning,
								  inputs=[GAME, gamma, learning_rate, epsilon, lambd, max_steps, log_episode],
								  num_episodes=num_episodes, learning_times=learning_times, save=True,
								  title=results_dir + GAME + "_q_eligibility.txt")

	if args.learning_type == "macro_q_imitation":
		num_g_train_traces = 10
		g_type = "sequitur"
		k = 2
		results= average_results(func=imitation_macro_learning,
								 inputs=[GAME, gamma, learning_rate, epsilon, 0, max_steps, num_g_train_traces, warm_runs, g_type, k, log_episode],
								 num_episodes=num_episodes, learning_times=learning_times, save=True,
								 title=results_dir + GAME + "_imitation_2seq.txt")

		results= average_results(func=imitation_macro_learning,
								 inputs=[GAME, gamma, learning_rate, epsilon, lambd, max_steps, num_g_train_traces, warm_runs, g_type, k, log_episode],
								 num_episodes=num_episodes, learning_times=learning_times, save=True,
								 title=results_dir + GAME + "_imitation_2seq_eligibility.txt")

	if args.learning_type == "macro_q_online":
		num_g_train_traces = 10
		warm_runs = 50
		num_updates = 10
		g_type = "sequitur"
		k = 2
		results = average_results(func=online_macro_learning,
								  inputs=[GAME, gamma, learning_rate, epsilon, 0, max_steps, num_g_train_traces, warm_runs, g_type, k, num_updates, False, log_episode],
								  num_episodes=num_episodes, learning_times=learning_times, save=True,
								  title=results_dir + GAME + "_online_2seq.txt", grammar=True)

		results = average_results(func=online_macro_learning,
								  inputs=[GAME, gamma, learning_rate, epsilon, lambd, max_steps, num_g_train_traces, warm_runs, g_type, k, num_updates, False, log_episode],
								  num_episodes=num_episodes, learning_times=learning_times, save=True,
								  title=results_dir + GAME + "_online_2seq_eligibility.txt", grammar=True)

		results = average_results(func=online_macro_learning,
								  inputs=[GAME, gamma, learning_rate, epsilon, 0, max_steps, num_g_train_traces, warm_runs, g_type, k, num_updates, True, log_episode],
								  num_episodes=num_episodes, learning_times=learning_times, save=True,
								  title=results_dir + GAME + "_online_transfer_2seq.txt", grammar=True)

		results = average_results(func=online_macro_learning,
								  inputs=[GAME, gamma, learning_rate, epsilon, lambd, max_steps, num_g_train_traces, warm_runs, g_type, k, num_updates, True, log_episode],
								  num_episodes=num_episodes, learning_times=learning_times, save=True,
								  title=results_dir + GAME + "_online_transfer_2seq_eligibility.txt", grammar=True)

	if args.learning_type == "options_imitation_hmm":
		warm_runs = 10000
		num_g_train_traces = 1000
		results = average_results(func=imitiation_options_learning,
								  inputs=[GAME, gamma, learning_rate, epsilon, max_steps, num_g_train_traces, warm_runs, [2, 10, ["hmm", 2]], log_episode],
								  num_episodes=num_episodes, learning_times=learning_times, save=True,
								  title=results_dir + GAME + "_imitation_options_hmm_1.txt")

		results = average_results(func=imitiation_options_learning,
								  inputs=[GAME, gamma, learning_rate, epsilon, max_steps, num_g_train_traces, warm_runs, [2, 10, ["hmm", 4]], log_episode],
								  num_episodes=num_episodes, learning_times=learning_times, save=True,
								  title=results_dir + GAME + "_imitation_options_hmm_2.txt")

		results = average_results(func=imitiation_options_learning,
								  inputs=[GAME, gamma, learning_rate, epsilon, max_steps, num_g_train_traces, warm_runs, [2, 10, ["hmm", 6]], log_episode],
								  num_episodes=num_episodes, learning_times=learning_times, save=True,
								  title=results_dir + GAME + "_imitation_options_hmm_3.txt")


	if args.learning_type == "options_imitation_lstm":
		warm_runs = 10000
		num_g_train_traces = 1000
		results = average_results(func=imitiation_options_learning,
								  inputs=[GAME, gamma, learning_rate, epsilon, max_steps, num_g_train_traces, warm_runs, [2, 10, ["rnn", "lstm", 1]], log_episode],
								  num_episodes=num_episodes, learning_times=learning_times, save=True,
								  title=results_dir + GAME + "_imitation_options_lstm_1.txt")

		results = average_results(func=imitiation_options_learning,
								  inputs=[GAME, gamma, learning_rate, epsilon, max_steps, num_g_train_traces, warm_runs, [2, 10, ["rnn", "lstm", 2]], log_episode],
								  num_episodes=num_episodes, learning_times=learning_times, save=True,
								  title=results_dir + GAME + "_imitation_options_lstm_2.txt")

		results = average_results(func=imitiation_options_learning,
								  inputs=[GAME, gamma, learning_rate, epsilon, max_steps, num_g_train_traces, warm_runs, [2, 10, ["rnn", "lstm", 3]], log_episode],
								  num_episodes=num_episodes, learning_times=learning_times, save=True,
								  title=results_dir + GAME + "_imitation_options_lstm_3.txt")

	if args.learning_type == "options_imitation_gru":
		warm_runs = 10000
		num_g_train_traces = 1000
		results = average_results(func=imitiation_options_learning,
								  inputs=[GAME, gamma, learning_rate, epsilon, max_steps, num_g_train_traces, warm_runs, [2, 10, ["rnn", "gru", 1]], log_episode],
								  num_episodes=num_episodes, learning_times=learning_times, save=True,
								  title=results_dir + GAME + "_imitation_options_gru_1.txt")

		results = average_results(func=imitiation_options_learning,
								  inputs=[GAME, gamma, learning_rate, epsilon, max_steps, num_g_train_traces, warm_runs, [2, 10, ["rnn", "gru", 1]], log_episode],
								  num_episodes=num_episodes, learning_times=learning_times, save=True,
								  title=results_dir + GAME + "_imitation_options_gru_2.txt")

		results = average_results(func=imitiation_options_learning,
								  inputs=[GAME, gamma, learning_rate, epsilon, max_steps, num_g_train_traces, warm_runs, [2, 10, ["rnn", "gru", 1]], log_episode],
								  num_episodes=num_episodes, learning_times=learning_times, save=True,
								  title=results_dir + GAME + "_imitation_options_gru_3.txt")

	if args.learning_type == "options_online_hmm":
		num_g_train_traces = 100
		warm_runs = 50
		num_updates = 10
		results = average_results(func=online_options_learning,
								  inputs=[GAME, gamma, learning_rate, epsilon, max_steps, num_g_train_traces, warm_runs, 1, [2, 10, ["hmm", 2]], num_updates, log_episode],
								  num_episodes=num_episodes, learning_times=learning_times, save=True,
								  title=results_dir + GAME + "_online_options_hmm_1.txt")

		results = average_results(func=online_options_learning,
								  inputs=[GAME, gamma, learning_rate, epsilon, max_steps, num_g_train_traces, warm_runs, 1, [2, 10, ["hmm", 4]], num_updates, log_episode],
								  num_episodes=num_episodes, learning_times=learning_times, save=True,
								  title=results_dir + GAME + "_online_options_hmm_2.txt")

		results = average_results(func=online_options_learning,
								  inputs=[GAME, gamma, learning_rate, epsilon, max_steps, num_g_train_traces, warm_runs, 1, [2, 10, ["hmm", 6]], num_updates, log_episode],
								  num_episodes=num_episodes, learning_times=learning_times, save=True,
								  title=results_dir + GAME + "_online_options_hmm_3.txt")


	if args.learning_type == "options_online_lstm":
		num_g_train_traces = 100
		warm_runs = 50
		num_updates = 10
		results = average_results(func=online_options_learning,
								  inputs=[GAME, gamma, learning_rate, epsilon, max_steps, num_g_train_traces, warm_runs, 1, [2, 10, ["rnn", "lstm", 1]], num_updates, log_episode],
								  num_episodes=num_episodes, learning_times=learning_times, save=True,
								  title=results_dir + GAME + "_online_options_lstm_1.txt")

		results = average_results(func=online_options_learning,
								  inputs=[GAME, gamma, learning_rate, epsilon, max_steps, num_g_train_traces, warm_runs, 1, [2, 10, ["rnn", "lstm", 2]], num_updates, log_episode],
								  num_episodes=num_episodes, learning_times=learning_times, save=True,
								  title=results_dir + GAME + "_online_options_lstm_2.txt")

		results = average_results(func=online_options_learning,
								  inputs=[GAME, gamma, learning_rate, epsilon, max_steps, num_g_train_traces, warm_runs, 1, [2, 10, ["rnn", "lstm", 3]], num_updates, log_episode],
								  num_episodes=num_episodes, learning_times=learning_times, save=True,
								  title=results_dir + GAME + "_online_options_lstm_3.txt")

	if args.learning_type == "options_online_gru":
		num_g_train_traces = 100
		warm_runs = 50
		num_updates = 10
		results = average_results(func=online_options_learning,
								  inputs=[GAME, gamma, learning_rate, epsilon, max_steps, num_g_train_traces, warm_runs, 1, [2, 10, ["rnn", "gru", 1]], num_updates, log_episode],
								  num_episodes=num_episodes, learning_times=learning_times, save=True,
								  title=results_dir + GAME + "_online_options_gru_1.txt")

		results = average_results(func=online_options_learning,
								  inputs=[GAME, gamma, learning_rate, epsilon, max_steps, num_g_train_traces, warm_runs, 1, [2, 10, ["rnn", "gru", 1]], num_updates, log_episode],
								  num_episodes=num_episodes, learning_times=learning_times, save=True,
								  title=results_dir + GAME + "_online_options_gru_2.txt")

		results = average_results(func=online_options_learning,
								  inputs=[GAME, gamma, learning_rate, epsilon, max_steps, num_g_train_traces, warm_runs, 1, [2, 10, ["rnn", "gru", 1]], num_updates, log_episode],
								  num_episodes=num_episodes, learning_times=learning_times, save=True,
								  title=results_dir + GAME + "_online_options_gru_3.txt")
