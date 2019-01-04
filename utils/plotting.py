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


def smooth_results(results, smooth, subset):
    """
    Output: Computes moving average for different cols of results array
    """
    avg_steps = moving_average(results[results[:, 0] < subset, 1], smooth)
    avg_steps_sd = moving_average(results[results[:, 0] < subset, 2], smooth)
    avg_ret = moving_average(results[results[:, 0] < subset, 3], smooth)
    avg_ret_sd = moving_average(results[results[:, 0] < subset, 4], smooth)
    return avg_steps, avg_steps_sd, avg_ret, avg_ret_sd


def plot_learning(results, smooth, no_rows, subset,
                  labels, steps_titles, reward_titles,
                  title, ylim_steps, ylim_rew, macros=None):
        episodes = results[0][results[0][:, 0] < subset, 0]

        step_avg = []
        step_sd = []
        ret_avg = []
        ret_sd = []

        no_plots = len(steps_titles) + len(reward_titles)
        for res in results:
            avg_ret, avg_ret_sd, avg_steps, avg_steps_sd = smooth_results(res, smooth, subset)
            step_avg.append(avg_steps)
            step_sd.append(avg_steps_sd)
            ret_avg.append(avg_ret)
            ret_sd.append(avg_ret_sd)


        fig, axs = plt.subplots(no_rows, no_plots/no_rows, figsize=(12, 12),
                                facecolor='w', edgecolor='k')
        axs = axs.ravel()
        fig.subplots_adjust(hspace=.3, wspace=.3, top=0.9)
        plt.suptitle(title, fontsize=16)

        for i in range(len(steps_titles)):
            axs_temp = axs[i]
            axs_temp.set_title(steps_titles[i])
            axs_temp.set_ylim(ylim_steps[0], ylim_steps[1])
            for j in range(len(labels)/2):
                axs_temp.plot(episodes, step_avg[len(labels)/2*i + j], CB_color_cycle[j], label=labels[len(labels)/2*i + j])
                axs_temp.plot(episodes, step_avg[len(labels)/2*i + j] + step_sd[len(labels)/2*i + j], CB_color_cycle[j], alpha=0.25)
                axs_temp.plot(episodes, step_avg[len(labels)/2*i + j] - step_sd[len(labels)/2*i + j], CB_color_cycle[j], alpha=0.25)
                axs_temp.fill_between(episodes, step_avg[len(labels)/2*i + j] - step_sd[len(labels)/2*i + j], step_avg[len(labels)/2*i + j] + step_sd[len(labels)/2*i + j],
                                      facecolor=CB_color_cycle[j], alpha=0.25)

            # axs_temp.set_ylim(bottom=-200, top=1100)
            axs_temp.set_xlabel('Episodes')
            axs_temp.set_ylabel('Steps to Goal')
            axs_temp.legend(loc="upper left", prop={'size': 6})

            if i > 0:
                width = 10
                axs_macro = axs_temp.twinx()

                eps = macros[0][macros[0][:, 0] < subset, 0]

                for i, updates in enumerate(macros):
                    macro_means = updates[updates[:, 0] < subset, 1]
                    macro_std = updates[updates[:, 0] < subset, 2]
                    axs_macro.bar(eps+width*i, macro_means, width, color=CB_color_cycle[i], yerr=macro_std)

                axs_macro.set_ylabel("Number of Extracted Macros")

        for i in range(len(reward_titles)):
            axs_temp = axs[len(reward_titles) + i]
            axs_temp.set_title(reward_titles[i])
            axs_temp.set_ylim(ylim_rew[0], ylim_rew[1])
            for j in range(len(labels)/2):
                axs_temp.plot(episodes, ret_avg[len(labels)/2*i + j], CB_color_cycle[j], label=labels[len(labels)/2*i + j])
                axs_temp.plot(episodes, ret_avg[len(labels)/2*i + j] + ret_sd[len(labels)/2*i + j], CB_color_cycle[j], alpha=0.25)
                axs_temp.plot(episodes, ret_avg[len(labels)/2*i + j] - ret_sd[len(labels)/2*i + j], CB_color_cycle[j], alpha=0.25)
                axs_temp.fill_between(episodes, ret_avg[len(labels)/2*i + j] - ret_sd[len(labels)/2*i + j], ret_avg[len(labels)/2*i + j] + ret_sd[len(labels)/2*i + j],
                                      facecolor=CB_color_cycle[j], alpha=0.25)

            # axs_temp.set_ylim(bottom=-0.15, top=0.75)
            axs_temp.set_xlabel('Episodes')
            axs_temp.set_ylabel('Rewards')
            axs_temp.legend(loc="upper left", prop={'size': 6})

            if i > 0:
                width = 10
                axs_macro = axs_temp.twinx()

                eps = macros[0][macros[0][:, 0] < subset, 0]

                for i, updates in enumerate(macros):
                    macro_means = updates[updates[:, 0] < subset, 3]
                    macro_std = updates[updates[:, 0] < subset, 4]
                    axs_macro.bar(eps+width*i, macro_means, width, color=CB_color_cycle[i], yerr=macro_std)

                axs_macro.set_ylabel("Length of Extracted Macros")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        return


def plot_options_learning(results, smooth, no_rows, subset,
                          labels, steps_titles,
                          title, ylim, update_eps=None):
        episodes = results[0][results[0][:, 0] < subset, 0]

        step_avg = []
        step_sd = []
        ret_avg = []
        ret_sd = []

        no_plots = len(steps_titles)
        no_cols = no_plots/no_rows

        for res in results:
            avg_ret, avg_ret_sd, avg_steps, avg_steps_sd = smooth_results(res, smooth, subset)
            step_avg.append(avg_steps)
            step_sd.append(avg_steps_sd)
            ret_avg.append(avg_ret)
            ret_sd.append(avg_ret_sd)


        fig, axs = plt.subplots(no_rows, no_cols, figsize=(12, 12),
                                facecolor='w', edgecolor='k')
        axs = axs.ravel()
        fig.subplots_adjust(hspace=.3, wspace=.3, top=0.9)
        plt.suptitle(title, fontsize=16)

        counter = 0
        for i in range(len(steps_titles)):
            axs_temp = axs[i]
            axs_temp.set_title(steps_titles[i])
            axs_temp.set_ylim(ylim[0], ylim[1])
            for j in range(len(labels)/no_cols):
                axs_temp.plot(episodes, step_avg[len(labels)/no_cols*i + j], CB_color_cycle[j], label=labels[counter])
                axs_temp.plot(episodes, step_avg[len(labels)/no_cols*i + j] + step_sd[len(labels)/no_cols*i + j], CB_color_cycle[j], alpha=0.25)
                axs_temp.plot(episodes, step_avg[len(labels)/no_cols*i + j] - step_sd[len(labels)/no_cols*i + j], CB_color_cycle[j], alpha=0.25)
                axs_temp.fill_between(episodes, step_avg[len(labels)/no_cols*i + j] - step_sd[len(labels)/no_cols*i + j], step_avg[len(labels)/no_cols*i + j] + step_sd[len(labels)/no_cols*i + j],
                                      facecolor=CB_color_cycle[j], alpha=0.25)

                counter += 1
                if counter == 9:
                    counter = 0
            # axs_temp.set_ylim(bottom=-200, top=1100)
            axs_temp.set_xlabel('Episodes')
            axs_temp.set_ylabel('Steps to Goal')
            axs_temp.legend(loc="upper left", prop={'size': 6})

            if i > (no_cols-1):
                update_eps = update_eps[update_eps < subset]
                for upd in update_eps:
                    axs_temp.axvline(x=upd, linestyle="--", alpha=0.5)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        return

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-env', '--environment', default="Taxi-v2",
    					help="RL environment")
    parser.add_argument('-sm', '--smooth', default=1, type=int,
    					help="Alpha for moving average")
    parser.add_argument('-s', '--save', action="store_true", default=True,
    					help="Alpha for moving average")

    args = parser.parse_args()
    GAME = args.environment
    smooth = args.smooth
    save = args.save

    base_dir = os.getcwd()
    results_dir = base_dir + "/results/"
    results_options_dir = base_dir + "/results_options/"
    figures_dir = base_dir + "/figures/"

    if GAME == "Taxi-v2":
        title = "Taxi Environment"
        ylim_rew = [-130, 20]
        ylim_steps = [10, 120]
        ylim = [0, 120]
    elif GAME == "FrozenLake-v0":
        title = "Frozen Lake Environment"
        ylim_rew = [-130, 20]
        ylim_steps = [10, 120]
        ylim = [0, 120]
    elif GAME == "FrozenLake8x8-v0":
        title = "Frozen Lake 8x8 Environment"
        ylim_rew = [-0.01, 0.03]
        ylim_steps = [0, 100]
        ylim = [0, 100]

    labels = ['Q-Learning', 'Q($\lambda$)-Learning',
              'SMDP-Q-Learning', 'SMDP-Q($\lambda$)-Learning',
              'SMDP-Q-Learning: No Transfer', 'SMDP-Q($\lambda$)-Learning: No Transfer',
              'SMDP-Q-Learning: Transfer', 'SMDP-Q($\lambda$)-Learning: Transfer']
    steps_titles = ['2-Sequitur Imitation CFAG Macros: Steps to Goal', '2-Sequitur Online CFAG Macros: Steps to Goal']
    reward_titles = ['2-Sequitur Imitation CFAG Macros: Reward', '2-Sequitur Online CFAG Macros: Reward']

    results_q = np.loadtxt(results_dir + GAME + '_q.txt')
    results_q_eligibility = np.loadtxt(results_dir + GAME + '_q_eligibility.txt')
    results_imitation = np.loadtxt(results_dir + GAME + '_imitation_2seq.txt')
    results_imitation_eligibility = np.loadtxt(results_dir + GAME + '_imitation_2seq_eligibility.txt')

    results_online = np.loadtxt(results_dir + GAME + '_online_2seq.txt')
    results_online_eligibility = np.loadtxt(results_dir + GAME + '_online_2seq_eligibility.txt')
    results_online_transfer = np.loadtxt(results_dir + GAME + '_online_transfer_2seq.txt')
    results_online_transfer_eligibility = np.loadtxt(results_dir + GAME + '_online_transfer_2seq_eligibility.txt')


    macros_online = np.loadtxt(results_dir + GAME + '_online_2seq_grammar.txt')
    macros_online_eligibility = np.loadtxt(results_dir + GAME + '_online_2seq_eligibility_grammar.txt')
    macros_online_transfer = np.loadtxt(results_dir + GAME + '_online_transfer_2seq_grammar.txt')
    macros_online_transfer_eligibility = np.loadtxt(results_dir + GAME + '_online_transfer_2seq_eligibility_grammar.txt')

    results = [results_q, results_q_eligibility,
               results_imitation, results_imitation_eligibility,
               results_online, results_online_eligibility,
               results_online_transfer, results_online_transfer_eligibility]

    macros = [macros_online, macros_online_eligibility,
              macros_online_transfer, macros_online_transfer_eligibility]

    smooth = 2

    plot_learning(results, smooth, 2, 1200,
                  labels, steps_titles, reward_titles, title,
                  ylim_steps, ylim_rew, macros)

    if save:
        plt.savefig(figures_dir + "results_" + GAME, dpi=300)

    ####################
    results_imitation_hmm_1 = np.loadtxt(results_options_dir + GAME + '_imitation_options_hmm_1.txt')
    results_imitation_hmm_2 = np.loadtxt(results_options_dir + GAME + '_imitation_options_hmm_2.txt')
    results_imitation_hmm_3 = np.loadtxt(results_options_dir + GAME + '_imitation_options_hmm_3.txt')

    results_imitation_lstm_1 = np.loadtxt(results_options_dir + GAME + '_imitation_options_lstm_1.txt')
    results_imitation_lstm_2 = np.loadtxt(results_options_dir + GAME + '_imitation_options_lstm_2.txt')
    results_imitation_lstm_3 = np.loadtxt(results_options_dir + GAME + '_imitation_options_lstm_3.txt')

    results_imitation_gru_1 = np.loadtxt(results_options_dir + GAME + '_imitation_options_gru_1.txt')
    results_imitation_gru_2 = np.loadtxt(results_options_dir + GAME + '_imitation_options_gru_2.txt')
    results_imitation_gru_3 = np.loadtxt(results_options_dir + GAME + '_imitation_options_gru_3.txt')

    results_online_hmm_1 = np.loadtxt(results_options_dir + GAME + '_online_options_hmm_1.txt')
    results_online_hmm_2 = np.loadtxt(results_options_dir + GAME + '_online_options_hmm_2.txt')
    results_online_hmm_3 = np.loadtxt(results_options_dir + GAME + '_online_options_hmm_3.txt')

    results_online_lstm_1 = np.loadtxt(results_options_dir + GAME + '_online_options_lstm_1.txt')
    results_online_lstm_2 = np.loadtxt(results_options_dir + GAME + '_online_options_lstm_2.txt')
    results_online_lstm_3 = np.loadtxt(results_options_dir + GAME + '_online_options_lstm_3.txt')

    results_online_gru_1 = np.loadtxt(results_options_dir + GAME + '_imitation_options_gru_1.txt')
    results_online_gru_2 = np.loadtxt(results_options_dir + GAME + '_imitation_options_gru_2.txt')
    results_online_gru_3 = np.loadtxt(results_options_dir + GAME + '_imitation_options_gru_3.txt')

    results_opt = [results_imitation_hmm_1, results_imitation_hmm_2, results_imitation_hmm_3,
                   results_imitation_lstm_1, results_imitation_lstm_2, results_imitation_lstm_3,
                   results_imitation_gru_1, results_imitation_gru_2, results_imitation_gru_3,
                   results_online_hmm_1, results_online_hmm_2, results_online_hmm_3,
                   results_online_lstm_1, results_online_lstm_2, results_online_lstm_3,
                   results_online_gru_1, results_online_gru_2, results_online_gru_3]

    labels = ['HMM Option: 2 Hidden States', 'HMM Option: 4 Hidden States', 'HMM Option: 6 Hidden States',
              'LSTM Option: 5 Cells', 'LSTM Option: 10 Cells', 'LSTM Option: 15 Cells',
              'GRU Option: 5 Cells', 'GRU Option: 10 Cells', 'GRU Option: 15 Cells']

    steps_titles = ['HMM Imitation PCFAG Option: Steps to Goal',
                    'LSTM Imitation PCFAG Option: Steps to Goal',
                    'GRU Imitation PCFAG Option: Steps to Goal',
                    'HMM Online PCFAG Option: Steps to Goal',
                    'LSTM Online PCFAG Option: Steps to Goal',
                    'GRU Online PCFAG Option: Steps to Goal']

    update_eps = np.linspace(50, 2000, 11)

    plot_options_learning(results_opt, smooth, 2, 1200,
                          labels, steps_titles, title, ylim, update_eps)

    if save:
        plt.savefig(figures_dir + "results_online_" + GAME, dpi=300)
