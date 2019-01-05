import os
import numpy as np
import argparse
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.unicode'] = True

CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']

def moving_average(a, n=3):
    """
    Input I: Time series and smoothing degree
    Output: Computes moving average for a time series (rets/steps) and order n
    """
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret / n


def smooth_results(avg_steps, sd_steps, avg_ret, sd_ret, smooth):
    """
    Output: Computes moving average for different cols of results array
    """
    sm_avg_steps = moving_average(avg_steps, smooth)
    sm_sd_steps = moving_average(sd_steps, smooth)
    sm_avg_ret = moving_average(avg_ret, smooth)
    sm_sd_ret = moving_average(sd_ret, smooth)
    return sm_avg_steps, sm_sd_steps, sm_avg_ret, sm_sd_ret

def plot_learning(episodes, avg_steps, sd_steps, avg_ret, sd_ret):
    return
