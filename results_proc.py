import time
import numpy as np
import multiprocessing
from contextlib import contextmanager
from functools import partial


def average_results(func, inputs, num_episodes,
					learning_times=5, save=True, title="temp.txt", grammar=False):

	pool = multiprocessing.Pool()
	func_multi = partial(func, *inputs)

	start = time.time()
	results_multi = pool.map(func_multi, [num_episodes] * learning_times)
	results_avg = process_results(results_multi, title, save, grammar)
	print("Done with runnin {} workers after {} secs".format(learning_times, time.time() - start))
	return results_avg


def process_results(results, title, save, grammar=False):
    results_all = np.empty((0,5))
    for i in range(len(results)):
		results_all = np.append(results_all, results[i][1], axis=0)
    results_avg = []
    for x in sorted(np.unique(results_all[...,0])):
        results_avg.append([x,
                            np.average(results_all[np.where(results_all[...,0]==x)][...,1]),
                            np.average(results_all[np.where(results_all[...,0]==x)][...,2]),
                            np.average(results_all[np.where(results_all[...,0]==x)][...,3]),
                            np.average(results_all[np.where(results_all[...,0]==x)][...,4])])
    if grammar:
        results_all_grammar = np.empty((0,3))
        for i in range(len(results)):
            results_all_grammar = np.append(results_all_grammar, results[i][2], axis=0)
	    results_avg_grammar = []
	    for x in sorted(np.unique(results_all_grammar[...,0])):
	        results_avg_grammar.append([x,
	                            np.average(results_all_grammar[np.where(results_all_grammar[...,0]==x)][...,1]),
	                            np.std(results_all_grammar[np.where(results_all_grammar[...,0]==x)][...,1]),
								np.average(results_all_grammar[np.where(results_all_grammar[...,0]==x)][...,2]),
	                            np.std(results_all_grammar[np.where(results_all_grammar[...,0]==x)][...,2])])
        np.savetxt(title[:-4] + "_grammar.txt", np.array(results_avg_grammar))
    if save:
        np.savetxt(title, np.array(results_avg))
        print("Done saving results")
    return np.array(results_avg)
