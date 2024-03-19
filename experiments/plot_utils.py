import os
from os.path import join

import matplotlib.pyplot as plt
import seaborn as sns


def plot_varying(varying_elements, results_proposed, results_marginal, base_dir, experiment, mode, varying_range):
    """ Plots the ROC curve for a given true positive and false positive rate.
    """
    with sns.plotting_context("paper", rc={"font.size": 15, "axes.titlesize": 15,
                                           "axes.labelsize": 15, "legend.fontsize": 12,
                                           "lines.markersize": 8, "xtick.labelsize": 10,
                                           "ytick.labelsize": 10}):
        ax = sns.lineplot(x=varying_elements, y=results_proposed['mean'], errorbar=None, estimator=None, label='proposed', color='blue', marker = 'o')
        sns.lineplot(x=varying_elements, y=results_marginal['mean'], errorbar=None, estimator=None, label='marginal', color='orange', marker='o')
        plt.fill_between(varying_elements, results_proposed['mean'] - results_proposed['std'], results_proposed['mean'] + results_proposed['std'], alpha =0.2)
        plt.fill_between(varying_elements, results_marginal['mean'] - results_marginal['std'],
                         results_marginal['mean'] + results_marginal['std'], alpha=0.2)
        ax.set_title('Varying ' + str(mode) + ' curve');
        ax.set(xlabel='Noise ' + str(mode), ylabel='Loss');
        ax.legend(frameon=False)

        if isinstance(experiment, list):
            data_path = base_dir
        else:
            data_path = join(base_dir, experiment)

        plt.savefig(data_path + '/plot_varying_' + str(mode) + '_' + str(varying_range) + '.png', bbox_inches="tight", dpi=300)
        plt.close()