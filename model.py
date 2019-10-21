import pyro
import pyro.distributions as dist
import torch
import pandas as pd
import numpy as np
import os
import logging
from torch.distributions import constraints


def feature_generation(data):

    # Get station id dummies:
    station_onehot = pd.get_dummies(data['start_station_id'])
    station_onehot.columns = ['station_' +
                              str(c) for c in station_onehot.columns]

    features = {
        'demand': torch.tensor(
            data['demand'].values,
            dtype=torch.float),
        'stations': torch.tensor(
            station_onehot.values,
            dtype=torch.float)}

    feature_names = {'stations': station_onehot.columns.values}

    return features, feature_names


def model(stations, demand):

    coef = {}
    for s in feature_names['stations']:
        coef[s] = pyro.sample(s, dist.Normal(0, 1))

    log_lmbda = 0
    for i, s in enumerate(feature_names['stations']):
        log_lmbda += coef[s] * stations[:, i]

    lmbda = log_lmbda.exp()

    with pyro.plate("data", len(demand)):
        y = pyro.sample("obs", dist.Poisson(lmbda), obs=demand)

    return y


def guide(stations, demand):

    weights_loc = pyro.param('weights_loc', torch.randn(stations.shape[1]))
    weights_scale = pyro.param('weights_scale', torch.ones(stations.shape[1]),
                               constraint=constraints.positive)

    coef = {}
    for i, s in enumerate(feature_names['stations']):
        coef[s] = pyro.sample(s, dist.Normal(weights_loc[i], weights_scale[i]))

    log_lmbda = 0
    for i, s in enumerate(feature_names['stations']):
        log_lmbda += coef[s] * stations[:, i]

    lmbda = log_lmbda.exp()


if __name__ == '__main__':

    import pickle
    from pyro.infer import EmpiricalMarginal, SVI, Trace_ELBO, JitTrace_ELBO
    import pyro.optim as optim
    logging.basicConfig(format='%(message)s', level=logging.INFO)
    # Enable validation checks
    pyro.enable_validation(True)
    smoke_test = ('CI' in os.environ)
    pyro.set_rng_seed(1)

    with open('data/demand.pickle', 'rb') as f:
        data1 = pickle.load(f)

    features, feature_names = feature_generation(data1.sample(frac = .01))


    pyro.clear_param_store()
    svi = SVI(model,
              guide,
              optim.Adam({"lr": .005}),
              loss=JitTrace_ELBO(),
              num_samples=1000)

    num_iters = 5000 if not smoke_test else 2
    for i in range(num_iters):
        elbo = svi.step(features['stations'], features['demand'])
        if i % 500 == 0:
            logging.info("Elbo loss: {}".format(elbo))

    svi_diagnorm_posterior = svi.run(features['stations'], features['demand'])

    svi_samples = {s: EmpiricalMarginal(svi_diagnorm_posterior, sites=s)
                   .enumerate_support().detach().cpu().numpy()
                   for s in feature_names['stations']}

    from pyro.infer.mcmc.api import MCMC
    from pyro.infer.mcmc import NUTS
    nuts_kernel = NUTS(model)

    mcmc = MCMC(nuts_kernel, num_samples=1000, warmup_steps=200)
    mcmc_run = mcmc.run(features['stations'], features['demand'])