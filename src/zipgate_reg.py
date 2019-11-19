# Zero Inflated Poisson Regression
# Rate Estimate: lambda = exp(station + hour)
# Gate Estimate: rho = sigmoid(station + hour)

import numpy as np
import pandas as pd
import pyro
import pyro.distributions as dist
import torch
from torch.distributions import constraints
from torch import sigmoid


def feature_generation(data):
    # Get station id dummies:
    station_onehot = pd.get_dummies(
        data['start_station_id']).add_prefix('station_')

    # Get hour dummies
    hour_onehot = pd.get_dummies(data['hour']).add_prefix('hour_')

    # Weekday
    data['daytype'] = np.where(data['weekday'] < 5, 'weekday', 'weekend')
    daytype_onehot = pd.get_dummies(data['daytype'])

    # Feature df
    feature_df = pd.concat([station_onehot, hour_onehot, daytype_onehot],
                           axis=1)

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
                                 for c in hour_onehot.columns])},
                    'daytype': {'names': daytype_onehot.columns.values,
                                'index': np.array([
                                    feature_df.columns.get_loc(c)
                                    for c in daytype_onehot.columns])}}

    return data, feature_info


class ZIPoissRegGate:

    def __init__(self, features, data):

        self.features = features

    def model(self, data, demand):
        coef = {}

        for s in self.features['station']['names']:
            coef[s] = pyro.sample(s, dist.Normal(0, 1))
            s += '_gate'
            coef[s] = pyro.sample(s, dist.Normal(0, 1))

        for h in self.features['hour']['names']:
            for d in self.features['daytype']['names']:
                name = h + '_' + d
                coef[name] = pyro.sample(name, dist.Normal(0, 1))
                name += '_gate'
                coef[name] = pyro.sample(name, dist.Normal(0, 1))

        log_lmbda = 0
        gate_mean = 0
        for i in range(len(self.features['station']['names'])):
            name = self.features['station']['names'][i]
            index = self.features['station']['index'][i]
            log_lmbda += coef[name] * data[:, index]
            gate_mean += coef[name + '_gate'] * data[:, index]

        for h in range(len(self.features['hour']['names'])):
            for d in range(len(self.features['daytype']['names'])):
                h_name = self.features['hour']['names'][h]
                h_index = self.features['hour']['index'][h]
                d_name = self.features['daytype']['names'][d]
                d_index = self.features['daytype']['index'][d]
                log_lmbda += coef[h_name + '_' + d_name] * \
                    data[:, h_index] * data[:, d_index]

                gate_mean += coef[h_name + '_' + d_name + '_gate'] * \
                    data[:, h_index] * data[:, d_index]

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

        station_gate_loc = pyro.param('station_gate_loc',
                                      torch.randn(n_stations))
        station_gate_scale = pyro.param('station_gate_scale',
                                        torch.ones(n_stations),
                                        constraint=constraints.positive)

        n_hours = len(self.features['hour']['names'])
        n_daytype = len(self.features['daytype']['names'])
        hour_daytype_loc = pyro.param('hour_dattype_loc',
                                      torch.randn(n_hours * n_daytype))
        hour_daytype_scale = pyro.param('hour_dattype_scale',
                                        torch.ones(n_hours * n_daytype),
                                        constraint=constraints.positive)

        hour_daytype_gate_loc = pyro.param('hour_dattype_gate_loc',
                                           torch.randn(n_hours * n_daytype))
        hour_daytype_gate_scale = pyro.param('hour_dattype_gate_scale',
                                             torch.ones(n_hours * n_daytype),
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

            coef[name + '_gate'] = pyro.sample(name + '_gate',
                                               dist.Normal(station_gate_loc[i],
                                                           station_gate_scale[
                                                               i]))

            log_lmbda += coef[name] * data[:, index]
            gate_mean += coef[name + '_gate'] * data[:, index]

        for h in range(len(self.features['hour']['names'])):
            for d in range(len(self.features['daytype']['names'])):
                h_name = self.features['hour']['names'][h]
                h_index = self.features['hour']['index'][h]
                d_name = self.features['daytype']['names'][d]
                d_index = self.features['daytype']['index'][d]

                name = h_name + '_' + d_name
                i = h * n_daytype + d
                coef[name] = pyro.sample(name,
                                         dist.Normal(hour_daytype_loc[i],
                                                     hour_daytype_scale[i]))
                coef[name + '_gate'] = pyro.sample(name + '_gate', dist.Normal(
                    hour_daytype_gate_loc[i], hour_daytype_gate_scale[i]))

                log_lmbda += coef[h_name + '_' + d_name] * \
                    data[:, h_index] * data[:, d_index]
                gate_mean += coef[h_name + '_' + d_name + '_gate'] * \
                    data[:, h_index] * data[:, d_index]

        lmbda = log_lmbda.exp()
        gate = sigmoid(gate_mean)

        return gate, lmbda

    def wrapped_model(self, data, demand):
        # https://pyro.ai/examples/bayesian_regression.html#Inference
        gate, lmbda = self.model(data, demand)
        pyro.sample("gate_post", dist.Delta(gate))
        pyro.sample("lmbda_post", dist.Delta(lmbda))


if __name__ == '__main__':
    pass
