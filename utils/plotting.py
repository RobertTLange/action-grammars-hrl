import os
import numpy as np
import argparse
import matplotlib.pyplot as plt

CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']

cols_to_plot = ["#B31329","#1386A0"]


def smooth(ts, windowSize):
    # Perform smoothed moving average with specified window to time series
    ts_MA = []
    weights = np.repeat(1.0, windowSize) / windowSize
    if type(ts) != list:
        ts = list(ts)
    for i in range(len(ts)):
        ts_MA.append(np.convolve(ts[i], weights, 'valid'))
    return ts_MA


def plot_learning(episodes, med_ts, p10_ts, p90_ts, smooth_degree,
                  title, label_temp, grammar_update_ind=None,
                  save_fname=None):
    med_ts = smooth(med_ts, smooth_degree)
    p10_ts = smooth(p10_ts, smooth_degree)
    p90_ts = smooth(p90_ts, smooth_degree)

    fig = plt.figure(figsize=(10, 10))
    plot = fig.add_subplot(111)

    for i in range(len(med_ts)):
        plt.plot(episodes[i][smooth_degree-1:], med_ts[i], cols_to_plot[i],
                 label=label_temp[i], linewidth=3)
        plt.plot(episodes[i][smooth_degree-1:], p10_ts[i],
                 cols_to_plot[i], alpha=0.25, linewidth=2)
        plt.plot(episodes[i][smooth_degree-1:], p90_ts[i],
                 cols_to_plot[i], alpha=0.25, linewidth=2)
        plt.fill_between(episodes[i][smooth_degree-1:], p10_ts[i], p90_ts[i],
                         facecolor=cols_to_plot[i], alpha=0.25)

    if grammar_update_ind is not None:
        for i, xc in enumerate(grammar_update_ind):
            if i == len(grammar_update_ind) - 1:
                plt.axvline(x=xc, linestyle='--', alpha=0.7,
                             label="Grammar Update")
            else:
                plt.axvline(x=xc, linestyle='--', alpha=0.7)

    plt.legend(loc="upper right", fontsize=20)
    plt.title(title, fontsize=25)
    plt.xlabel("Episodes", fontsize=20)
    plt.ylabel("Steps until Goal", fontsize=20)

    plot.tick_params(axis='both', which='major', labelsize=20)
    plot.tick_params(axis='both', which='minor', labelsize=8)
    plt.tight_layout()
    
    if save_fname is not None:
        plt.savefig(save_fname, dpi=900)
        print("Saved figure to {}".format(save_fname))
    return


def plot_all_learning(its, steps, sd_steps, rew, sd_rew,
                      smooth_degree, labels):
    plt.figure(figsize=(15, 12), dpi=200)

    counter = 0

    for i in range(len(its)):
        counter += 2
        plt.subplot(len(its), 2, counter-1)
        plot_learning(its[i], steps[i], sd_steps[i],
                      smooth_degree, str(i+4) + " Disks: Steps to Goal",
                      labels)

        plt.subplot(len(its), 2, counter)
        plot_learning(its[i], rew[i], sd_rew[i],
                      smooth_degree, str(i+4) + " Disks: Discounted Reward",
                      labels)
