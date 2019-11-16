## Defining the Model

use_cuda = torch.cuda.is_available()

k = 2
alpha = 1
a, b = 1, 1

def model(data):
    global k, alpha, a, b
    global use_cuda
    
    pi = pyro.sample("pi", distcc.Dirichlet(alpha * torch.ones(k)))
    if use_cuda:
        pi = pi.cuda()
    
    with pyro.plate("components", k):
        mean = pyro.sample(
            "mean", distcc.MultivariateNormal(torch.zeros(2), torch.eye(2))
        )
        sigma = pyro.sample("sigma", distcc.InverseGamma(a, b))
        
        if use_cuda:
            mean = mean.cuda()
            sigma = sigma.cuda()
    
    with pyro.plate("data", len(data) if data is not None else _n*2):
        z = pyro.sample("lv", distcc.Categorical(pi))
        
        if use_cuda:
            z = z.cuda()
        
        _sigma = sigma[z].repeat(2, 2, 1).transpose(2, 0)
        
        if use_cuda:
            _sigma *= torch.eye(2).repeat(len(z), 1, 1).cuda()
        else:
            _sigma *= torch.eye(2).repeat(len(z), 1, 1)
                    
        obs = pyro.sample(
            "obs", distcc.MultivariateNormal(mean[z], _sigma), obs=data
        )
    
    return obs
