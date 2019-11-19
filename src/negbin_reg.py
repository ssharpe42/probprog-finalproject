# Zero Inflated Poisson Regression
# Rate Estimate: exp(station + hour*daytype)
# Gate Estimate: global Beta

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


class NegBinReg:

    def __init__(self, features, data):

        self.features = features

    def model(self, data, demand):
        coef = {}

        for s in self.features['station']['names']:
            coef[s] = pyro.sample(s, dist.Normal(0, 1))
            s += '_count'
            coef[s] = pyro.sample(s, dist.Normal(0, 1))

        for h in self.features['hour']['names']:
            for d in self.features['daytype']['names']:
                name = h + '_' + d
                coef[name] = pyro.sample(name, dist.Normal(0, 1))
                name+='_count'
                coef[name] = pyro.sample(name, dist.Normal(0, 1))

        logits = 0
        count_mean = 0
        for i in range(len(self.features['station']['names'])):
            name = self.features['station']['names'][i]
            index = self.features['station']['index'][i]
            logits += coef[name] * data[:, index]
            count_mean+=coef[name+'_count'] * data[:, index]

        for h in range(len(self.features['hour']['names'])):
            for d in range(len(self.features['daytype']['names'])):
                h_name = self.features['hour']['names'][h]
                h_index = self.features['hour']['index'][h]
                d_name = self.features['daytype']['names'][d]
                d_index = self.features['daytype']['index'][d]
                logits += coef[h_name + '_' + d_name] * \
                    data[:, h_index] * data[:, d_index]
                count_mean += coef[h_name + '_' + d_name + '_count'] * \
                    data[:, h_index] * data[:, d_index]

        prob = sigmoid(logits)
        total_count = count_mean.exp()

        with pyro.plate("data", len(data)):
            pyro.sample(
                "obs", dist.NegativeBinomial (
                    total_count, prob), obs=demand)

            return total_count, prob

    def guide(self, data, demand):

        n_stations = len(self.features['station']['names'])
        station_w_loc = pyro.param('station_w_loc', torch.randn(n_stations))
        station_w_scale = pyro.param('station_w_scale', torch.ones(n_stations),
                                     constraint=constraints.positive)

        station_count_loc = pyro.param('station_count_loc',
                                      torch.randn(n_stations))
        station_count_scale = pyro.param('station_count_scale',
                                        torch.ones(n_stations),
                                        constraint=constraints.positive)

        n_hours = len(self.features['hour']['names'])
        n_daytype = len(self.features['daytype']['names'])
        hour_daytype_loc = pyro.param('hour_dattype_loc',
                                      torch.randn(n_hours * n_daytype))
        hour_daytype_scale = pyro.param('hour_dattype_scale',
                                        torch.ones(n_hours * n_daytype),
                                        constraint=constraints.positive)

        hour_daytype_count_loc = pyro.param('hour_dattype_count_loc',
                                           torch.randn(n_hours * n_daytype))
        hour_daytype_count_scale = pyro.param('hour_dattype_count_scale',
                                             torch.ones(n_hours * n_daytype),
                                             constraint=constraints.positive)

        coef = {}
        logits = 0
        count_mean=0
        for i in range(len(self.features['station']['names'])):
            name = self.features['station']['names'][i]
            index = self.features['station']['index'][i]

            coef[name] = pyro.sample(name,
                                     dist.Normal(station_w_loc[i],
                                                 station_w_scale[i]))

            coef[name + '_count'] = pyro.sample(name + '_count',
                                               dist.Normal(station_count_loc[i],
                                                           station_count_scale[
                                                               i]))

            logits += coef[name] * data[:, index]
            count_mean += coef[name + '_count'] * data[:, index]

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

                coef[name+ '_count'] = pyro.sample(name+ '_count',
                                         dist.Normal(hour_daytype_count_loc[i],
                                                    hour_daytype_count_scale[i]))

                logits += coef[h_name + '_' + d_name] * \
                    data[:, h_index] * data[:, d_index]
                count_mean += coef[h_name + '_' + d_name+ '_count'] * \
                    data[:, h_index] * data[:, d_index]


        prob = sigmoid(logits)
        total_count = count_mean.exp()

        return total_count, prob

    def wrapped_model(self, data, demand):

        total_count, prob = self.model(data, demand)
        pyro.sample("total_count_post", dist.Delta(total_count))
        pyro.sample("prob_post", dist.Delta(prob))


