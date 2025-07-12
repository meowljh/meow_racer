import torch
from torch.distributions import MultivariateNormal, categorical

def extract(input, t:torch.Tensor, x:torch.Tensor):
    if t.ndim == 0:
        t = t.unsqueeze(0)
    shape = x.shape 
    t = t.long().to(input.device)
    out = torch.gather(input, 0, t)
    reshape = [t.shape[0]] + [1] * (len(shape) - 1) #(B, 1 for _ in range(len(shape)-1))

    return out.reshape(*reshape)


class DifferentiableGMM:
    def __init__(self, weights, means, covariances, device):
        self.device = device
        self.weights = torch.tensor(weights, dtype=torch.float32, device=self.device)
        self.means = torch.tensor(means, dtype=torch.float32, device=self.device)
        self.covs = torch.stack([
            torch.tensor(cov, dtype=torch.float32, device=self.device) for cov in covariances
        ])
        self.components = [
            MultivariateNormal(loc=self.means[i], covariance_matrix=self.covs[i])
            for i in range(len(weights))
        ]

    def log_prob(self, x):  # x: (N, D), requires_grad OK
        log_probs = []
        for i, component in enumerate(self.components):
            log_pi = torch.log(self.weights[i] + 1e-8)  # avoid log(0)
            log_p = component.log_prob(x)              # (N,)
            log_probs.append(log_pi + log_p)
        log_probs = torch.stack(log_probs, dim=1)       # (N, K)
        return torch.logsumexp(log_probs, dim=1)        # (N,)

    def prob(self, x):
        return torch.exp(self.log_prob(x))              # (N,)

    def score_fn(self, x):
        """Returns ∇_x log p(x)"""
        x = x.clone().detach().requires_grad_(True)
        logp = self.log_prob(x)
        grad = torch.autograd.grad(logp.sum(), x)[0]    # ∇_x log p(x)

        return grad