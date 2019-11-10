# Zero Inflated Poisson Regression

import numpy as np
import pandas as pd
import pyro
import pyro.distributions as dist
import torch
from torch.distributions import constraints

def model2_data(data):
  # RE-create the data matrix to suit the new model by multiplying the 
  # column_station_i with column_hour_j
  stations = 70
  hours = 24
  X = data
  N = X.shape[0] # number of samples of data
  M = stations*hours # number of regressors that we have
  X_new = torch.zeros(N,M)

  count = 0
  for i in range(stations):
      for j in range(stations,94,1):

          xij = torch.mul(X[:,i],X[:,j])
          X_new[:,count] = xij
          count +=1

  # Sanity check that my new X matrix should have exactly one 1 in each 
  # row due to the multiplication of column_i and column_j
  # for i in range(N):
  #     assert(list(X_new[i,:]).count(1)==1),"I expect exactly one 1 in each row, which didn't happen for row {}".format(i)

  return X_new


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


class station_hr_ZIP_model:

    def __init__(self, features, data):

        self.features = features

        
    def model(self,data,demand):
        X_new = model2_data(data)
        coef = {}
        count = 0
        log_lmbda = 0
        for s1 in self.features['station']['names']:
            for s2 in self.features['hour']['names']:
                s = s1+s2
                coef[s] = pyro.sample(s, dist.Normal(0, 1))
                log_lmbda += coef[s] * X_new[:,count]
                count+=1

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
        # print(data)
        X_new = model2_data(data)
        data = X_new
        weights_loc = pyro.param('weights_loc', torch.randn(data.shape[1]))
        weights_scale = pyro.param('weights_scale', torch.ones(data.shape[1]),
                                   constraint=constraints.positive)

        gate_alpha = pyro.param('gate_alpha', torch.tensor(3.),
                                constraint=constraints.positive)
        gate_beta = pyro.param('gate_beta', torch.tensor(3.),
                               constraint=constraints.positive)

        coef = {}
        log_lmbda = 0

        count = 0
        for s1 in self.features['station']['names']:
            for s2 in self.features['hour']['names']:
                s = s1+s2
                coef[s] = pyro.sample(s, dist.Normal(weights_loc[count], weights_scale[count]))
                log_lmbda += coef[s] * X_new[:,count]
                count+=1
                
        lmbda = log_lmbda.exp()
        gate = pyro.sample('gate', dist.Beta(gate_alpha, gate_beta))


    def wrapped_model(self, data, demand):
        # This shouldn't be delta in this case like
        # https://pyro.ai/examples/bayesian_regression.html#Inference
        # pyro.sample("prediction", dist.Delta(model(data, demand)))
        # We want a poisson output
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
