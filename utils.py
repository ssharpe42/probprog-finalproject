import pandas as pd
from pyro.infer import EmpiricalMarginal


def svi_samples(svi_posterior, sites):

    return {site: EmpiricalMarginal(svi_posterior, sites=site)
            .enumerate_support().detach().cpu().numpy()
            for site in sites}


def mcmc_samples(mcmc):

    return {k: v.detach().cpu().numpy() for k, v in mcmc.get_samples().items()}


def summary(samples):
    """Summarize posterior samples"""

    site_stats = {}
    for site_name, values in samples.items():
        marginal_site = pd.DataFrame(values)

        describe = marginal_site.describe(
            percentiles=[.05, 0.25, 0.5, 0.75, 0.95]).transpose()

        site_stats[site_name] = describe[[
            "mean", "std", "5%", "25%", "50%", "75%", "95%"]]
    return site_stats
