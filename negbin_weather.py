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


    # Get mean temperature 
    mean_temp = data[['mean_temperature_f']]/120
    mean_temp_squared = mean_temp**2
    constant_ones = np.ones(len(data['mean_temperature_f']))
    constant_ones = pd.DataFrame(data=constant_ones)

    # Feature df
    feature_df = pd.concat([station_onehot, hour_onehot, daytype_onehot, mean_temp, mean_temp_squared, constant_ones],
                           axis=1)

    data = {
        'demand': torch.tensor(
            data['demand'].values,
            dtype=torch.float),
        'data': torch.tensor(
            feature_df.values,
            dtype=torch.float)}
    
    i = feature_df.columns.get_loc(daytype_onehot.columns[-1])
    

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
                                    for c in daytype_onehot.columns])},
                    'temperature_f':{'names': np.array(['mean_temp','mean_temp_squared', 'ones']),
                                     'index': np.array([i+1,i+2,i+3])}}

    return data, feature_info


class NegBinReg:

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

        coef['mean_temp'] = pyro.sample('mean_temp', dist.Normal(0, 1))
        coef['mean_temp_squared'] = pyro.sample('mean_temp_squared', dist.Normal(0, 1))
        coef['ones'] = pyro.sample('ones', dist.Normal(0, 1))

        logits = 0
        for i in range(len(self.features['station']['names'])):
            name = self.features['station']['names'][i]
            index = self.features['station']['index'][i]
            logits += coef[name] * data[:, index]

        for h in range(len(self.features['hour']['names'])):
            for d in range(len(self.features['daytype']['names'])):
                h_name = self.features['hour']['names'][h]
                h_index = self.features['hour']['index'][h]
                d_name = self.features['daytype']['names'][d]
                d_index = self.features['daytype']['index'][d]
                logits += coef[h_name + '_' + d_name] * \
                    data[:, h_index] * data[:, d_index]

        logits += coef['mean_temp'] * data[:,-3] # linear term
        logits += coef['mean_temp_squared'] * data[:,-2] # quadratic term
        logits += coef['ones'] * data[:,-1] #constant

        eps = 0.01
        prob = sigmoid(logits) 
        p = prob.clone()

        # for i in range(len(p)):
        #     if prob[i] ==1:
        #         p[i]= prob[i]-eps

        # for i in range(len(p)):
        #     if prob[i] ==0:
        #         p[i]= prob[i]+eps

        print("probs are",p)

        total_count = pyro.sample('total_count', dist.Gamma(1, 1))

        with pyro.plate("data", len(data)):
            pyro.sample(
                "obs", dist.NegativeBinomial (
                    total_count, p), obs=demand)

            return total_count, p

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

        n_temp = len(self.features['temperature_f']['names'])
        temp_loc = pyro.param('temp_loc', torch.randn(n_temp))
        temp_scale = pyro.param('temp_scale',torch.ones(n_temp),
                                constraint=constraints.positive)

        total_count_loc = pyro.param('total_count_loc', torch.tensor(5.),
                                    constraint=constraints.positive)

        coef = {}
        logits = 0
        for i in range(len(self.features['station']['names'])):
            name = self.features['station']['names'][i]
            index = self.features['station']['index'][i]

            coef[name] = pyro.sample(name,
                                     dist.Normal(station_w_loc[i],
                                                 station_w_scale[i]))

            logits += coef[name] * data[:, index]

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

                logits += coef[h_name + '_' + d_name] * \
                    data[:, h_index] * data[:, d_index]

        for i in range(len(self.features['temperature_f']['names'])):
            name = self.features['temperature_f']['names'][i] # mean_temp, mean_temp_squared, ones
            print(name)
            index = self.features['temperature_f']['index'][i] # used to index data

            coef[name] = pyro.sample(name,
                                     dist.Normal(temp_loc[i],
                                                 temp_scale[i]))
            logits += coef[name] * data[:, index]

        total_count = pyro.sample('total_count', dist.Normal(total_count_loc,
                                                           torch.tensor(0.5)))
        
        prob = sigmoid(logits) 

        print("guide prob is",prob)
        return total_count, prob

    def wrapped_model(self, data, demand):
        total_count, prob = self.model(data, demand)
        pyro.sample("total_count_post", dist.Delta(total_count))
        pyro.sample("prob_post", dist.Delta(prob))
