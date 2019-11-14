# Zero Inflated Poisson Regression
# Rate Estimate: exp(station + hour*daytype)
# Gate Estimate: global Beta

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


class ZIPoissReg:

    def __init__(self, features, data):

        self.features = features

    def model(self, data, demand):
        coef = {}

        for s in self.features['station']['names']:
            coef[s] = pyro.sample(s, dist.Normal(0, 1))

        for h in self.features['hour']['names']:
            for d in self.features['daytype']['names']:
                name = h + '_' + d
                coef[name] = pyro.sample(name, dist.Normal(0, 1))

        log_lmbda = 0
        for i in range(len(self.features['station']['names'])):
            name = self.features['station']['names'][i]
            index = self.features['station']['index'][i]
            log_lmbda += coef[name] * data[:, index]

        for h in range(len(self.features['hour']['names'])):
            for d in range(len(self.features['daytype']['names'])):
                h_name = self.features['hour']['names'][h]
                h_index = self.features['hour']['index'][h]
                d_name = self.features['daytype']['names'][d]
                d_index = self.features['daytype']['index'][d]
                log_lmbda += coef[h_name + '_' + d_name] * \
                    data[:, h_index] * data[:, d_index]

        lmbda = log_lmbda.exp()

        gate_alpha = pyro.sample('gate_alpha', dist.Gamma(2, 2))
        gate_beta = pyro.sample('gate_beta', dist.Gamma(3, 2))
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
        n_daytype = len(self.features['daytype']['names'])
        hour_daytype_loc = pyro.param('hour_dattype_loc',
                                      torch.randn(n_hours * n_daytype))
        hour_daytype_scale = pyro.param('hour_dattype_scale',
                                        torch.ones(n_hours * n_daytype),
                                        constraint=constraints.positive)

        gate_alpha_loc = pyro.param('gate_alpha_loc', torch.tensor(3.),
                                    constraint=constraints.positive)
        gate_beta_loc = pyro.param('gate_beta_loc', torch.tensor(3.),
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

                log_lmbda += coef[h_name + '_' + d_name] * \
                    data[:, h_index] * data[:, d_index]

        gate_alpha = pyro.sample('gate_alpha', dist.Normal(gate_alpha_loc,
                                                           torch.tensor(0.05)))
        gate_beta = pyro.sample('gate_beta', dist.Normal(gate_beta_loc,
                                                         torch.tensor(0.05)))

        lmbda = log_lmbda.exp()
        gate = pyro.sample('gate', dist.Beta(gate_alpha, gate_beta))

        return gate, lmbda

    def wrapped_model(self, data, demand):
        # https://pyro.ai/examples/bayesian_regression.html#Inference
        gate, lmbda = self.model(data, demand)
        pyro.sample("gate_post", dist.Delta(gate))
        pyro.sample("lmbda_post", dist.Delta(lmbda))


if __name__ == '__main__':
    pass
    # import pickle
    #
    # with open('data/demand_3h.pickle', 'rb') as f:
    #     data1 = pickle.load(f)
    #
    # samp = data1.groupby(['start_station_id', 'hour']).apply(
    #     lambda x: x.sample(100)).reset_index(drop=True)
    # with open('data/demand_sample.pickle', 'wb') as f:
    #     pickle.dump(samp, f)
