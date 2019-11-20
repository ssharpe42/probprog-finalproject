import os
import logging
import pyro
from pyro.infer import SVI, JitTrace_ELBO
import pyro.optim as optim

pyro.enable_validation(True)
smoke_test = ('CI' in os.environ)
pyro.set_rng_seed(1)
logging.basicConfig(format='%(message)s', level=logging.INFO)


def ewma_step(current, new, alpha=.9):
    """ Exponential moving avg visualizing loss"""

    return current * (1 - alpha) + new * alpha


def run_svi(model, guide, iters, data, demand, num_samples=1000, filename=''):
    """
    Runs SVI

    :param model: pyro model
    :param guide: pyro guide
    :param iters: iterations
    :param data: data to be passed to model, guide
    :param demand: demand to be passed to model, guide
    :param num_samples: number of samples for Monte Carlo posterior
                        approximation
    :param filename: file to save pyro param store (.pkl)
    :return: svi object, and elbo loss
    """

    pyro.clear_param_store()
    svi = SVI(model,
              guide,
              optim.Adam({"lr": .005}),
              loss=JitTrace_ELBO(),
              num_samples=num_samples)

    num_iters = iters if not smoke_test else 2

    elbo_losses = []
    for i in range(num_iters):
        elbo = svi.step(data, demand)
        elbo_losses.append(elbo)
        if i % 1000 == 0:
            logging.info("Elbo loss: {}".format(elbo))

    if filename:
        pyro.get_param_store().save(filename)

    return svi, elbo_losses


def get_svi_posterior(data, demand, svi=None, model=None,
                      guide=None,
                      num_samples=100,
                      filename=''):
    """
    Extract posterior

    :param data: data to be passed to model, guide
    :param demand: demand to be passed to model, guide
    :param svi: svi object
    :param model: pyro model
    :param guide: pyro guide
    :param num_samples: number of samples to generate
    :param filename: param store to load
    :return: posterior
    """

    if svi is None and filename and model and guide:
        pyro.get_param_store().load(filename)

        svi = SVI(model,
                  guide,
                  optim.Adam({"lr": .005}),
                  loss=JitTrace_ELBO(),
                  num_samples=num_samples)

        svi.run(data, demand)

        return svi
    elif svi:
        svi.run(data, demand)
        return svi
    else:
        raise ValueError('Provide svi object or model/guide and filename')
