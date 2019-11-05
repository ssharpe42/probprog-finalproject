# Zero Inflated Poisson Regression
# Gate also estimated as regression

import numpy as np
import pandas as pd
import pyro
import pyro.distributions as dist
import torch
from torch.distributions import constraints
from torch.nn.functional import sigmoid


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


class ZIPoissRegGate:

    def __init__(self, features, data):

        self.features = features

    def model(self, data, demand):
        coef = {}

        for s in self.features['station']['names']:
            coef[s] = pyro.sample(s, dist.Normal(0, 1))

        for s in self.features['station']['names']:
            s+='_gate'
            coef[s] = pyro.sample(s, dist.Normal(0, 1))

        for s in self.features['hour']['names']:
            coef[s] = pyro.sample(s, dist.Normal(0, 1))

        log_lmbda = 0
        gate_mean = 0
        for i in range(len(self.features['station']['names'])):
            name = self.features['station']['names'][i]
            index = self.features['station']['index'][i]
            log_lmbda += coef[name] * data[:, index]
            gate_mean += coef[name+'_gate'] * data[:, index]

        for i in range(len(self.features['hour']['names'])):
            name = self.features['hour']['names'][i]
            index = self.features['hour']['index'][i]
            log_lmbda += coef[name] * data[:, index]

        lmbda = log_lmbda.exp()
        gate = sigmoid(gate_mean)

        with pyro.plate("data", len(data)):
            pyro.sample(
                "obs", dist.ZeroInflatedPoisson(
                    gate, lmbda), obs=demand)

            return gate, lmbda

    def guide(self, data, demand):

        n_stations = len(self.features['station']['names'])
        station_w_loc = pyro.param('station_w_loc', torch.randn(n_stations))
        station_w_scale = pyro.param('station_w_scale', torch.ones(n_stations),
                                   constraint=constraints.positive)

        station_gate_loc = pyro.param('station_gate_loc', torch.randn(n_stations))
        station_gate_scale = pyro.param('station_gate_scale', torch.ones(n_stations),
                                   constraint=constraints.positive)


        n_hours = len(self.features['hour']['names'])
        hour_w_loc = pyro.param('hour_w_loc', torch.randn(n_hours))
        hour_w_scale = pyro.param('hour_w_scale', torch.ones(n_hours),
                                   constraint=constraints.positive)

        coef = {}
        log_lmbda = 0
        gate_mean = 0
        for i in range(len(self.features['station']['names'])):
            name = self.features['station']['names'][i]
            index = self.features['station']['index'][i]

            coef[name] = pyro.sample(name,
                                     dist.Normal(station_w_loc[i],
                                                 station_w_scale[i]))

            coef[name+'_gate'] = pyro.sample(name+'_gate',
                                     dist.Normal(station_gate_loc[i],
                                                 station_gate_scale[i]))

            log_lmbda += coef[name] * data[:, index]
            gate_mean += coef[name+'_gate'] * data[:, index]

        for i in range(len(self.features['hour']['names'])):
            name = self.features['hour']['names'][i]
            index = self.features['hour']['index'][i]

            coef[name] = pyro.sample(name,
                                     dist.Normal(hour_w_loc[i],
                                                 hour_w_scale[i]))

            log_lmbda += coef[name] * data[:, index]

        lmbda = log_lmbda.exp()
        gate = sigmoid(gate_mean)

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
