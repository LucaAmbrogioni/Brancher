import warnings

import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_posterior(model, variables, number_samples=2000):

    # Get samples
    sample = model.get_sample(number_samples)
    post_sample = model.get_posterior_sample(number_samples)

    # Join samples
    sample["Mode"] = "Prior"
    post_sample["Mode"] = "Posterior"
    subsample = sample[variables + ["Mode"]]
    post_subsample = post_sample[variables + ["Mode"]]
    joint_subsample = subsample.append(post_subsample)

    # Plot posterior
    warnings.filterwarnings('ignore')
    g = sns.PairGrid(joint_subsample, hue="Mode")
    g = g.map_offdiag(sns.kdeplot)
    g = g.map_diag(sns.kdeplot, lw=3, shade=True)
    g = g.add_legend()
    warnings.filterwarnings('default')


def plot_density(model, variables, number_samples=2000):
    sample = model.get_sample(number_samples)
    warnings.filterwarnings('ignore')
    g = sns.PairGrid(sample[variables])
    g = g.map_offdiag(sns.kdeplot)
    g = g.map_diag(sns.kdeplot, lw=3, shade=True)
    g = g.add_legend()
    warnings.filterwarnings('default')


def ensemble_histogram(sample_list, variable, weights, bins=30):
    num_samples = sum([len(s) for s in sample_list])
    num_resamples = [int(np.ceil(w*num_samples*2)) for w in weights]
    max_samples = max(num_resamples)
    hist_df = pd.DataFrame()
    for idx, s in enumerate(sample_list):
        num_remaining_samples = max_samples - num_resamples[idx]
        resampled_values = np.concatenate([s[variable].sample(num_resamples[idx], replace=True).values,
                                           np.array([np.nan]*num_remaining_samples)])
        hist_df["Model {}".format(idx)] = resampled_values
    hist_df.plot.hist(stacked=True, bins=bins)


def plot_particles(particles, var_name, dim1, dim2, **kwargs):
    x, y = [[p.get_variable("weights").value.detach().numpy().flatten()[dim] for p in particles] for dim in [dim1, dim2]]
    plt.scatter(x, y, **kwargs)

