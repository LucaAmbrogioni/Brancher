import warnings

import seaborn as sns
import pandas as pd


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


def ensemble_histogram(sample_list, variable, bins=30):
    hist_df = pd.DataFrame()
    for idx, s in enumerate(sample_list):
        hist_df["Model {}".format(idx)] = s[variable]
    hist_df.plot.hist(stacked=True, bins=bins)


