# Zero Inflated Poisson Regression

import numpy as np
import pandas as pd
import pyro
import pyro.distributions as dist
import torch
from torch.distributions import constraints


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


class ZIPoissReg:

    def __init__(self, features, data):

        self.features = features

    def model(self, data, demand):
        coef = {}

        for s in self.features['station']['names']:
            coef[s] = pyro.sample(s, dist.Normal(0, 1))

        for s in self.features['hour']['names']:
            coef[s] = pyro.sample(s, dist.Normal(0, 1))

        log_lmbda = 0
        for i in range(len(self.features['station']['names'])):
            name = self.features['station']['names'][i]
            index = self.features['station']['index'][i]
            log_lmbda += coef[name] * data[:, index]

        for i in range(len(self.features['hour']['names'])):
            name = self.features['hour']['names'][i]
            index = self.features['hour']['index'][i]
            log_lmbda += coef[name] * data[:, index]

        lmbda = log_lmbda.exp()

        gate_alpha = pyro.sample('gate_alpha_prior', dist.Gamma(2, .6))
        gate_beta = pyro.sample('gate_beta_prior', dist.Gamma(3, .4))
        gate = pyro.sample('gate', dist.Beta(gate_alpha, gate_beta))

        with pyro.plate("data", len(data)):
            pyro.sample(
                "obs", dist.ZeroInflatedPoisson(
                    gate, lmbda), obs=demand)

            # should we be returning lmbda?
            return gate, lmbda

    def guide(self, data, demand):

        n_stations = len(self.features['station']['names'])
        station_w_loc = pyro.param('station_w_loc', torch.randn(n_stations))
        station_w_scale = pyro.param('station_w_scale', torch.ones(n_stations),
                                   constraint=constraints.positive)

        n_hours = len(self.features['hour']['names'])
        hour_w_loc = pyro.param('hour_w_loc', torch.randn(n_hours))
        hour_w_scale = pyro.param('hour_w_scale', torch.ones(n_hours),
                                   constraint=constraints.positive)

        gate_alpha = pyro.param('gate_alpha', torch.tensor(3.),
                                constraint=constraints.positive)
        gate_beta = pyro.param('gate_beta', torch.tensor(3.),
                               constraint=constraints.positive)

        coef = {}
        log_lmbda = 0
        for i in range(len(self.features['station']['names'])):
            name = self.features['station']['names'][i]
            index = self.features['station']['index'][i]

            coef[name] = pyro.sample(name,
                                     dist.Normal(station_w_loc[i],
                                                 station_w_scale[i]))

            log_lmbda += coef[name] * data[:, index]

        for i in range(len(self.features['hour']['names'])):
            name = self.features['hour']['names'][i]
            index = self.features['hour']['index'][i]

            coef[name] = pyro.sample(name,
                                     dist.Normal(hour_w_loc[i],
                                                 hour_w_scale[i]))

            log_lmbda += coef[name] * data[:, index]

        lmbda = log_lmbda.exp()
        gate = pyro.sample('gate', dist.Beta(gate_alpha, gate_beta))

    def wrapped_model(self, data, demand):
        # This shouldn't be delta in this case like
        # https://pyro.ai/examples/bayesian_regression.html#Inference
        # pyro.sample("prediction", dist.Delta(model(data, demand)))
        pyro.sample(
            "prediction",
            dist.ZeroInflatedPoisson(
                *
                self.model(
                    data,
                    demand)))


if __name__ == '__main__':
    import pickle
    with open('data/demand_3h.pickle', 'rb') as f:
        data1 = pickle.load(f)

    samp = data1.groupby(['start_station_id', 'hour']).apply(
        lambda x: x.sample(100)).reset_index(drop=True)
    with open('data/demand_sample.pickle', 'wb') as f:
        pickle.dump(samp, f)
