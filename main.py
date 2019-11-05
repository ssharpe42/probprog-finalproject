import pickle

from baseline import *
#from pois_reg_sine import *
from zipreg_model import *
from zipgate_reg import *
from inference import *
from criticism import *
from preprocess_bikeshare import *




def main():

    with open('data/demand_sample.pickle', 'rb') as f:
        data_samp = pickle.load(f)

    # Split into train and test
    train = data_samp.sample(frac=0.5, random_state=42)
    test = data_samp.drop(train.index)


    #global features
    data, features = feature_generation(train)
    p = PoissReg(features, data)
    #p = ZIPoissReg(features, data)
    #p = ZIPoissRegGate(features, data)

    train_new = False

    if train_new:
        svi, elbo_loss = run_svi(p.model, p.guide,
                                 iters=5000,
                                 data=data['data'],
                                 demand=data['demand'],
                                 filename='models/svi_baseline_params.pkl')

        plot_elbo(elbo_loss)

        svi_posterior = get_svi_posterior(data['data'], data['demand'],
                                          svi=svi)

        print(svi.information_criterion())

    else:
        svi_posterior = get_svi_posterior(data['data'], data['demand'],
                                        model = p.model,
                                          guide = p.guide,
                                          filename='models/svi_zip_params.pkl')

    post_samples = posterior_samples(
        p.wrapped_model,
        svi_posterior,
        data,
        ['obs','prediction'],
        num_samples=1000)


    # post_samples = posterior_samples(
    #     p.model,
    #     svi_posterior,
    #     data,
    #     ['gate'],
    #     num_samples=1000)


    # post_samples = posterior_samples(
    #     p.model,
    #     svi_posterior,
    #     data,
    #     ['hour_amplitude','hour_period'],
    #     num_samples=200)

    # Example test statistics
    compare_test_statistic(test.demand.values, post_samples[:,1,:],
                           stat=perc_0)
    compare_test_statistic(test.demand.values, post_samples[:, 1, :],
                           stat=max_)
    compare_test_statistic(test.demand.values, post_samples[:, 1, :],
                           stat=percentile, q=80)



    station_info = get_station_info(conn =sqlite3.connect('data/sf_bikeshare.sqlite'))

    summary = site_summary(post_samples, ['obs','prediction'])





if __name__ == '__main__':
    main()
