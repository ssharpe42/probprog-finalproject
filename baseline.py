import logging
import os

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

# 2362.1897
class PoissReg:

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

        with pyro.plate("data", len(data)):
            y = pyro.sample("obs", dist.Poisson(lmbda), obs=demand)

            # should we be returning lmbda?
            return lmbda

    def guide(self, data, demand):
        weights_loc = pyro.param('weights_loc', torch.randn(data.shape[1]))
        weights_scale = pyro.param('weights_scale', torch.ones(data.shape[1]),
                                   constraint=constraints.positive)

        coef = {}
        log_lmbda = 0
        for i in range(len(self.features['station']['names'])):
            name = self.features['station']['names'][i]
            index = self.features['station']['index'][i]

            coef[name] = pyro.sample(name,
                                     dist.Normal(weights_loc[index],
                                                 weights_scale[index]))

            log_lmbda += coef[name] * data[:, index]

        for i in range(len(self.features['hour']['names'])):
            name = self.features['hour']['names'][i]
            index = self.features['hour']['index'][i]

            coef[name] = pyro.sample(name,
                                     dist.Normal(weights_loc[index],
                                                 weights_scale[index]))

            log_lmbda += coef[name] * data[:, index]

        lmbda = log_lmbda.exp()

    def wrapped_model(self, data, demand):
        # This shouldn't be delta in this case like https://pyro.ai/examples/bayesian_regression.html#Inference
        # pyro.sample("prediction", dist.Delta(model(data, demand)))
        # We want a poisson output
        pyro.sample("prediction", dist.Poisson(self.model(data, demand)))


if __name__ == '__main__':
    pass
    # import pickle
    # with open('data/demand_3h.pickle', 'rb') as f:
    #     data1 = pickle.load(f)
    #
    # samp = data1.groupby(['start_station_id','hour']).apply(lambda x: x.sample(300)).reset_index(drop = True)
    # with open('data/demand_sample.pickle','wb') as f:
    #     pickle.dump(samp, f)

