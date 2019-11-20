import os
from collections import defaultdict
import torch
from matplotlib import pyplot

import pyro
import pyro.distributions as dist
from pyro import poutine
from pyro.infer.autoguide import AutoDelta
from pyro.infer import SVI, TraceEnum_ELBO, config_enumerate, infer_discrete
smoke_test = ('CI' in os.environ)
pyro.enable_validation(True)

# Adapted from
# https://pyro.ai/examples/gmm.html#Predicting-membership-using-discrete-inference


def create_sites(station, features):
    sites_list = []
    for s in features['station']['names']:
        if s == station:
            for h in features['hour']['names']:
                for d in features['daytype']['names']:

                    name = s + '_' + h + '_' + d
                    sites_list.append(name)
    return sites_list


def run_GMM(data, K):

    @config_enumerate
    def model(data):
        # Global variables.
        weights = pyro.sample('weights', dist.Dirichlet(0.5 * torch.ones(K)))
        scale = pyro.sample('scale', dist.LogNormal(4., 2.))
        with pyro.plate('components', K):
            locs = pyro.sample('locs', dist.Normal(0., 10.))

        with pyro.plate('data', len(data)):
            # Local variables.
            assignment = pyro.sample('assignment', dist.Categorical(weights))
            pyro.sample('obs', dist.Normal(locs[assignment], scale), obs=data)

    optim = pyro.optim.Adam({'lr': 0.1, 'betas': [0.8, 0.99]})
    elbo = TraceEnum_ELBO(max_plate_nesting=1)

    def init_loc_fn(site):
        if site["name"] == "weights":
            # Initialize weights to uniform.
            return torch.ones(K) / K
        if site["name"] == "scale":
            return (data.var() / 2).sqrt()
        if site["name"] == "locs":
            return data[torch.multinomial(
                torch.ones(len(data)) / len(data), K)]
        raise ValueError(site["name"])

    def initialize(seed):
        global global_guide, svi
        pyro.set_rng_seed(seed)
        pyro.clear_param_store()
        global_guide = AutoDelta(
            poutine.block(
                model,
                expose=[
                    'weights',
                    'locs',
                    'scale']),
            init_loc_fn=init_loc_fn)
        svi = SVI(model, global_guide, optim, loss=elbo)
        return svi.loss(model, global_guide, data)

    # Choose the best among 100 random initializations.
    loss, seed = min((initialize(seed), seed) for seed in range(100))
    initialize(seed)
    print('seed = {}, initial_loss = {}'.format(seed, loss))

    # Register hooks to monitor gradient norms.
    gradient_norms = defaultdict(list)
    for name, value in pyro.get_param_store().named_parameters():
        value.register_hook(
            lambda g,
            name=name: gradient_norms[name].append(
                g.norm().item()))

    losses = []
    for i in range(200 if not smoke_test else 2):
        loss = svi.step(data)
        losses.append(loss)
        print('.' if i % 100 else '\n', end='')
    print()

    map_estimates = global_guide(data)
    weights = map_estimates['weights']
    locs = map_estimates['locs']
    scale = map_estimates['scale']
    print('weights = {}'.format(weights.data.numpy()))
    print('locs = {}'.format(locs.data.numpy()))
    print('scale = {}'.format(scale.data.numpy()))

    guide_trace = poutine.trace(global_guide).get_trace(
        data)  # record the globals
    trained_model = poutine.replay(
        model, trace=guide_trace)  # replay the globals

    def classifier(data, temperature=0):
        inferred_model = infer_discrete(
            trained_model,
            temperature=temperature,
            first_available_dim=-
            2)  # avoid conflict with data plate
        trace = poutine.trace(inferred_model).get_trace(data)
        return trace.nodes["assignment"]["value"]

    assignment = classifier(data)

    pyplot.figure(figsize=(8, 2), dpi=100).set_facecolor('white')
    pyplot.plot(data.numpy(), assignment.numpy(), 'bx')
    pyplot.title('MAP assignment')
    pyplot.xlabel('Latent posterior sample value')
    pyplot.ylabel('class assignment')

    return assignment


def assign_station_categories(assignment):
    stat_cat = []
    N_samples_per_beta = 100

    for i in range(0, len(assignment), N_samples_per_beta):
        assign = list(assignment[i:i + 100])
        a = max(set(assign), key=assign.count)
        stat_cat.append(a)
    return stat_cat


def create_dict(sites, station_categories):

    cat = []
    for i in station_categories:
        cat.append(i.item())

    latent = []
    for i in station_categories:
        name = 'category' + '_' + str(i.numpy())
        latent.append(name)

    dic = {key: [] for key in latent}
    for i, x in enumerate(cat):
        name = 'category' + '_' + str(x)
        dic[name].append(sites[i - 1])
    return dic


def tabulate_results(dic):
    print('Category\t\tLatent Variable')
    print('--------\t\t--------------')
    cnt = 0
    for key in dic:
        if dic[key] != []:
            name = str(cnt)
            cnt += 1
            latent_string = ', '.join([v[11:] for v in dic[key]])
            print(f'{name}\t\t{latent_string}')
