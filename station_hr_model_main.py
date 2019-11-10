import pickle

from station_hr_model import *
from inference import *
from criticism import *


with open('data/demand_sample.pickle', 'rb') as f:
    data_samp = pickle.load(f)


#global features
data, features = feature_generation(data_samp)

p = station_hr_ZIP_model(features, data)


svi_posterior = get_svi_posterior(data['data'], data['demand'],
                                model = p.model,
                                guide = p.guide,
                                filename='models/nihaar_model/svi_zip_params.pkl')


post_samples = posterior_samples(
    p.wrapped_model,
    svi_posterior,
    data,
    ['obs','prediction'],
    num_samples=80)


compare_test_statistic(data_samp.demand.values, post_samples[:,1,:],
                       stat=perc_0)