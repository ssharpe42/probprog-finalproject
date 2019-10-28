import pickle

from model import *
from inference import *
from criticism import *




def main():

    with open('data/demand_sample.pickle', 'rb') as f:
        data_samp = pickle.load(f)


    #global features
    data, features = feature_generation(data_samp)
    p = PoissReg(features, data)

    svi, elbo_loss = run_svi(p.model, p.guide,
                             iters=1000,
                             data=data['data'],
                             demand=data['demand'],
                             filename='models/svi_params.pkl')

    plot_elbo(elbo_loss)
    print(svi.information_criterion())

    svi_posterior = get_svi_posterior(data['data'], data['demand'], svi=svi)

    post_samples = posterior_samples(
        p.wrapped_model,
        svi_posterior,
        data,
        ['obs','prediction'],
        num_samples=200)

    summary = site_summary(post_samples, ['obs','prediction'])


if __name__ == '__main__':
    main()
