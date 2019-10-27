import logging
import os

import numpy as np
import pandas as pd
import pyro
import pyro.distributions as dist
import torch
from torch.distributions import constraints
from functools import partial


def feature_generation(data):

    # Get station id dummies:
    station_onehot = pd.get_dummies(
        data['start_station_id']).add_prefix('station_')

    # Get hour dummies
    hour_onehot = pd.get_dummies(data['hour']).add_prefix('hour_')

    # Feature df
    feature_df = pd.concat([station_onehot, hour_onehot], axis=1)

    data = {
        'demand': torch.tensor(
            data['demand'].values,
            dtype=torch.float),
        'data': torch.tensor(
            feature_df.values,
            dtype=torch.float)}

    feature_info = {'station': {'names': station_onehot.columns.values,
                                'index': np.array([
                                    feature_df.columns.get_loc(c)
                                    for c in station_onehot.columns])},
                    'hour': {'names': hour_onehot.columns.values,
                             'index': np.array([
                                 feature_df.columns.get_loc(c)
                                 for c in hour_onehot.columns])}}

    return data, feature_info


def model(data, demand):
    coef = {}
    for s in features['station']['names']:
        coef[s] = pyro.sample(s, dist.Normal(0, 1))

    for s in features['hour']['names']:
        coef[s] = pyro.sample(s, dist.Normal(0, 1))

    log_lmbda = 0
    for i in range(len(features['station']['names'])):
        name = features['station']['names'][i]
        index = features['station']['index'][i]
        log_lmbda += coef[name] * data[:, index]

    for i in range(len(features['hour']['names'])):
        name = features['hour']['names'][i]
        index = features['hour']['index'][i]
        log_lmbda += coef[name] * data[:, index]

    lmbda = log_lmbda.exp()

    with pyro.plate("data", len(data)):
        y = pyro.sample("obs", dist.Poisson(lmbda), obs=demand)

        # should we be returning lmbda?
        return y


def guide(data, demand):
    weights_loc = pyro.param('weights_loc', torch.randn(data.shape[1]))
    weights_scale = pyro.param('weights_scale', torch.ones(data.shape[1]),
                               constraint=constraints.positive)

    coef = {}
    log_lmbda = 0
    for i in range(len(features['station']['names'])):
        name = features['station']['names'][i]
        index = features['station']['index'][i]

        coef[name] = pyro.sample(
            name,
            dist.Normal(
                weights_loc[index],
                weights_scale[index]))
        log_lmbda += coef[name] * data[:, index]

    lmbda = log_lmbda.exp()


def wrapped_model(data, demand):
    # This shouldn't be delta in this case like https://pyro.ai/examples/bayesian_regression.html#Inference
    # pyro.sample("prediction", dist.Delta(model(data, demand)))
    # We want a poisson output
    pyro.sample("prediction", dist.Poisson(model(data, demand)))


if __name__ == '__main__':

    import pickle
    from pyro.infer import EmpiricalMarginal, SVI, JitTrace_ELBO, TracePredictive
    import pyro.optim as optim
    logging.basicConfig(format='%(message)s', level=logging.INFO)
    # Enable validation checks
    pyro.enable_validation(True)
    smoke_test = ('CI' in os.environ)
    pyro.set_rng_seed(1)
    pd.options.display.max_rows = 100
    pd.options.display.max_columns = 100

    # with open('data/demand.pickle', 'rb') as f:
    #     data1 = pickle.load(f)
    #
    # samp = data1.groupby(['start_station_id','hour']).apply(lambda x: x.sample(5)).reset_index(drop = True)
    # with open('data/demand_sample.pickle','wb') as f:
    #     pickle.dump(samp, f)

    with open('data/demand_sample.pickle', 'rb') as f:
        data_samp = pickle.load(f)

    data, features = feature_generation(data_samp)

    pyro.clear_param_store()
    svi = SVI(model,
              guide,
              optim.Adam({"lr": .005}),
              loss=JitTrace_ELBO(),
              num_samples=1000)

    num_iters = 3000 if not smoke_test else 2
    for i in range(num_iters):
        elbo = svi.step(data['data'], data['demand'])
        if i % 500 == 0:
            logging.info("Elbo loss: {}".format(elbo))

    # Posterior
    svi_posterior = svi.run(data['data'], data['demand'])
    # Posterior predictive
    trace_pred = TracePredictive(wrapped_model,
                                 svi_posterior,
                                 num_samples=100)
    post_pred = trace_pred.run(data['data'], None)

    def get_marginal(traces, sites): return EmpiricalMarginal(
        traces, sites)._get_samples_and_weights()[0].detach().cpu().numpy()

    # What is marginal in this case?
    marginal = get_marginal(post_pred, ['obs', 'prediction'])

    def summary(traces, sites):
        marginal = get_marginal(traces, sites)
        site_stats = {}
        for i in range(marginal.shape[1]):
            site_name = sites[i]
            marginal_site = pd.DataFrame(marginal[:, i]).transpose()
            describe = partial(pd.Series.describe,
                               percentiles=[.05, 0.25, 0.5, 0.75, 0.95])
            site_stats[site_name] = marginal_site.apply(
                describe, axis=1)[["mean", "std", "5%", "25%", "50%", "75%", "95%"]]
        return site_stats

    post_summary = summary(post_pred, sites=['prediction', 'obs'])

    pyro.get_param_store().save('models/svi_params.pkl')
    pyro.clear_param_store()
    pyro.get_param_store().load('models/svi_params.pkl')

    # Replay (https://forum.pyro.ai/t/tracepredictive-worse-than-sampling-guides/715/2)
    # preds = []
    # for _ in range(1000):
    #     guide_trace = guide(data['data'], None)
    #     preds.append(pyro.poutine.replay(model, guide_trace)(data['data'], None))

    # MCMC example
    # https://github.com/pyro-ppl/pyro/blob/dev/examples/baseball.py

    from pyro.infer.mcmc.api import MCMC
    from pyro.infer.mcmc import NUTS

    nuts_kernel = NUTS(model)

    mcmc = MCMC(nuts_kernel, num_samples=1000, warmup_steps=200)
    mcmc_run = mcmc.run(features['stations'], features['demand'])
