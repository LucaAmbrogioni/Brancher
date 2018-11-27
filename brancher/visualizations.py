import warnings

import seaborn as sns


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


