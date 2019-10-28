import os
import logging
import pyro
from pyro.infer import EmpiricalMarginal, SVI, JitTrace_ELBO, TracePredictive
import pyro.optim as optim

pyro.enable_validation(True)
smoke_test = ('CI' in os.environ)
pyro.set_rng_seed(1)
logging.basicConfig(format='%(message)s', level=logging.INFO)





def run_svi(model, guide, iters, data, demand, num_samples = 100, filename = ''):

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
        if i % 500 == 0:
            logging.info("Elbo loss: {}".format(elbo))

    if filename:
        pyro.get_param_store().save(filename)

    return svi, elbo_losses



def get_svi_posterior(data, demand, svi = None, model = None,
        guide = None,
        num_samples=100,
        filename = ''):

    if svi is None and filename and model and guide:
        pyro.get_param_store().load(filename)

        svi = SVI(model,
                  guide,
                  optim.Adam({"lr": .005}),
                  loss=JitTrace_ELBO(),
                  num_samples=num_samples)

        return svi.run(data, demand)
    elif svi:
        return svi.run(data, demand)
    else:
        raise ValueError('Provide svi object or model/guide and filename')

