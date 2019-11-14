# Some code taken from
# http://pyro.ai/examples/bayesian_regression.html#Model-Evaluation
# https://pyro.ai/examples/bayesian_regression_ii.html?highlight=guide
# https://pyro.ai/examples/bayesian_regression.html

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pyro.infer import EmpiricalMarginal, TracePredictive


# SVI Posterior Functions

def get_marginal(traces, sites):
    return (EmpiricalMarginal(traces, sites)
            ._get_samples_and_weights()[0]
            .detach().cpu().numpy())


def ppd_samples(
        wrapped_model,
        svi_posterior,
        data,
        sites,
        num_samples=200):
    """
    Get samples from posterior predictive

    :param wrapped_model: wrapped model with prediction site
    :param svi_posterior: posterior from svi.run
    :param data: covariate features
    :param sites: list of sites to take marginal over
    :param num_samples: number of samples from posterior
    :return: ppd
    """
    trace_pred = TracePredictive(wrapped_model,
                                 svi_posterior,
                                 num_samples=num_samples)
    post_pred = trace_pred.run(data['data'], None)
    marginal = get_marginal(post_pred, sites)

    return marginal


def posterior_site_samples(
        svi_posterior,
        sites):
    """
    Get site samples from posterior

    :param svi_posterior: posterior from svi.run
    :param sites: list of sites to take marginal over
    :return: marginal over sites
    """
    site_samples = {site: EmpiricalMarginal(svi_posterior,
                                            sites=site)
                    .enumerate_support().detach().cpu().numpy()
                    for site in sites}

    return site_samples


def site_summary(samples, sites):
    """
    Summarize posterior latent samples

    :param samples: dict of marginal samples from posterior_samples
    :param sites: sites to summarise
    :return: summary df
    """

    site_stats = {}
    for site_name in sites:
        values = samples[site_name]
        marginal_site = pd.DataFrame(values)
        describe = marginal_site.describe(
            percentiles=[.05, 0.25, 0.5, 0.75, 0.95]
        ).transpose()
        site_stats[site_name] = describe[[
            "mean", "std", "5%", "25%", "50%", "75%", "95%"]]

    return site_stats

# MCMC Posterior Functions


def mcmc_samples(mcmc):
    return {k: v.detach().cpu().numpy() for
            k, v in mcmc.get_samples().items()}


# Plotting Functions

def plot_elbo(elbo_losses):
    """Plot elbo loss"""

    fig, ax = plt.subplots()
    sns.lineplot(x=range(len(elbo_losses)),
                 y=elbo_losses,
                 ax=ax)


def compare_test_statistic(actual, predictive, stat=None, title='', **kwargs):
    """

    :param actual: numpy array of actual data
    :param predictive: numpy array of predictive distribution
                        expected shape (n samples, 1, data size)
    :param stat: test statistic function
    :param kwargs: extra arguments in test stat function
    :return: plot of comparison
    """

    actual_stat = stat(actual, **kwargs)
    pred_stat = stat(predictive, **kwargs)

    sns.distplot(pred_stat, kde=False)
    plt.axvline(actual_stat, 0, pred_stat.max(), color='red')
    plt.title(title)
    plt.show()


# Test statistics over n samples of a distribution

def mean(dist):
    dist = dist.squeeze()

    if len(dist.shape) == 1:
        return dist.mean()
    else:
        return dist.mean(axis=1)


def perc_0(dist):
    dist = dist.squeeze()

    if len(dist.shape) == 1:
        return (dist == 0).mean()
    else:
        return (dist == 0).mean(axis=1)


def max_(dist):
    dist = dist.squeeze()
    if len(dist.shape) == 1:
        return dist.max()
    else:
        return dist.max(axis=1)


def percentile(dist, q=99):
    dist = dist.squeeze()

    if len(dist.shape) == 1:
        return np.percentile(dist, q)
    else:
        return np.percentile(dist, q, axis=1)


def align_regressors_ppd(df, post_samples):
    """
    Align regressors in data with posterior samples

    :param df: original data
    :param post_samples: posterior samples
    :return: combined data
    """
    samples = pd.DataFrame(post_samples.T).add_prefix('sample_')
    comb_data = (pd.concat([df, samples], axis=1)
                 .reset_index()
                 .rename(columns={'index': 'samp_id'}))

    comb_data = (
        pd.melt(
            comb_data,
            id_vars=[
                'samp_id',
                'hour',
                'weekday',
                'start_station_id'],
            value_vars=comb_data.filter(
                regex='sample_').columns.values))

    return comb_data
