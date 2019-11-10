import pickle

#from model import *
#from model2 import *
from zipreg_model import *
from inference import *
from criticism import *



def main():

    with open('data/demand_sample.pickle', 'rb') as f:
        data_samp = pickle.load(f)

    train_new = True

    #global features
    data, features = feature_generation(data_samp)


    #p = PoissReg(features, data)
    p = ZIPoissReg(features, data)



    if train_new:
        svi, elbo_loss = run_svi(p.model, p.guide,
                                 iters=5000,
                                 data=data['data'],
                                 demand=data['demand'],
                                 filename='models/svi_zip_params_station_plus_hr.pkl')

        plot_elbo(elbo_loss)

        svi_posterior = get_svi_posterior(data['data'], data['demand'],
                                          svi=svi)

        print(svi.information_criterion())

    else:
        svi_posterior = get_svi_posterior(data['data'], data['demand'],
                                        model = p.model,
                                          guide = p.guide,
                                          filename='models/svi_params.pkl')

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
    compare_test_statistic(data_samp.demand.values, post_samples[:,1,:],
                           stat=perc_0)
    compare_test_statistic(data_samp.demand.values, post_samples[:, 1, :],
                           stat=max)
    compare_test_statistic(data_samp.demand.values, post_samples[:, 1, :],
                           stat=percentile, q=80)


    summary = site_summary(post_samples, ['obs','prediction'])





if __name__ == '__main__':
    main()
