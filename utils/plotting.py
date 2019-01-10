import os
import numpy as np
import argparse
import matplotlib.pyplot as plt

CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']

def smooth(ts, windowSize):
    # Perform smoothed moving average with specified window to time series
    ts_MA = []
    weights = np.repeat(1.0, windowSize) / windowSize
    if type(ts) != list:
        ts = list(ts)
    for i in range(len(ts)):
        ts_MA.append(np.convolve(ts[i], weights, 'valid'))
    return ts_MA


def plot_learning(episodes, mean_ts, sd_ts, smooth_degree, title, label_temp):
    mean_ts = smooth(mean_ts, smooth_degree)
    sd_ts = smooth(sd_ts, smooth_degree)

    for i in range(len(mean_ts)):
        plt.plot(episodes[smooth_degree-1:], mean_ts[i], CB_color_cycle[i], label=label_temp[i])
        plt.plot(episodes[smooth_degree-1:], mean_ts[i] - 2*sd_ts[i], CB_color_cycle[i], alpha=0.25)
        plt.plot(episodes[smooth_degree-1:], mean_ts[i] - 2*sd_ts[i], CB_color_cycle[i], alpha=0.25)
        plt.fill_between(episodes[smooth_degree-1:], mean_ts[i] - 2*sd_ts[i],
                         mean_ts[i] + 2*sd_ts[i],
                         facecolor=CB_color_cycle[i], alpha=0.25)
    plt.legend(loc=7)
    plt.title(title)
    return


def plot_all_learning(its, steps, sd_steps, rew, sd_rew,
                      smooth_degree, sub_titles, labels):
    plt.figure(figsize=(10, 8), dpi=200)

    counter = 0

    for i in range(len(its)):
        counter += 2
        plt.subplot(len(its), 2, counter-1)
        plot_learning(its[i], steps[i], sd_steps[i],
                      smooth_degree, sub_titles[counter-2],
                      labels)

        plt.subplot(len(its), 2, counter)
        plot_learning(its[i], rew[i], sd_rew[i],
                      smooth_degree, sub_titles[counter-1],
                      labels)
