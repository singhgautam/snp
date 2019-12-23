import math
import torch
import torch.nn.functional as F

def loss_recon_bernoulli_with_logit(logit, x):
    # p = recon prob
    return  F.binary_cross_entropy_with_logits(logit, x, size_average=False)

def loss_recon_bernoulli(p, x, reduction="full_sum"):
    if reduction == "full_sum":
        return torch.sum(F.binary_cross_entropy(p, x, reduce=False))
    elif reduction == "batch_sum":
        return torch.sum(F.binary_cross_entropy(p, x, reduce=False), dim=1)
    else:
        raise NotImplementedError

def loss_recon_gaussian(mu, logvar, x, const=None, reduction="full_sum"):
    # https://math.stackexchange.com/questions/1307381/logarithm-of-gaussian-function-is-whether-convex-or-nonconvex
    # mu, logvar = nomral distribution
    recon_loss_element = logvar + (x - mu)**2 / logvar.exp()

    # add const (can be used in change of variable)
    if const is not None:
        recon_loss_element += const

    recon_loss_element = recon_loss_element * 0.5

    if reduction == "full_sum":
        return torch.sum(recon_loss_element)
    elif reduction == "batch_sum":
        return torch.sum(recon_loss_element, dim=1)
    else:
        raise NotImplementedError

def loss_recon_gaussian_w_fixed_var(mu, x, std=1.0, const=None, reduction="full_sum", add_logvar=False):
    # init var, logvar
    var = std**2
    logvar = math.log(var)

    # estimate loss per element
    if add_logvar:
        recon_loss_element = logvar + (x - mu)**2 / var
    else:
        recon_loss_element = (x - mu)**2 / var

    # add const (can be used in change of variable)
    if const is not None:
        recon_loss_element += const

    recon_loss_element = recon_loss_element * 0.5

    if reduction == "full_sum":
        return torch.sum(recon_loss_element)
    elif reduction == "batch_sum":
        return torch.sum(recon_loss_element, dim=1)
    else:
        raise NotImplementedError


def loss_kld_gaussian(mu, logvar, do_sum=True):
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = 1 + logvar - mu.pow(2) - logvar.exp()

    # do sum
    if do_sum:
        KLD = torch.sum(KLD_element) * -0.5
        return KLD
    else:
        KLD_element = torch.sum(KLD_element, 1) * -0.5
        return KLD_element

def loss_kld_gaussian_vs_gaussian(mu1, logvar1, mu2, logvar2, do_sum=True):
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    # https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
    # log(sigma2) - log(sigma1) + 0.5 * (sigma1^2 + (mu1 - mu2)^2) / sigma2^2 - 0.5
    # 0 - log(sigma1) + 0.5 * (sigma1^2 + mu1^2)  - 0.5
    # 0 - log(sigma1) + 0.5 * sigma1^2 + 0.5 * mu1^2  - 0.5
    # 0 - 0.5 * log(sigma1^2) + 0.5 * sigma1^2 + 0.5 * mu1^2  - 0.5
    # log(sigma2) - log(sigma1) + 0.5 * (sigma1^2 + (mu1 - mu2)^2) / sigma2^2 - 0.5
    KLD_element = - logvar2 + logvar1 - (logvar1.exp() + (mu1 - mu2)**2) / logvar2.exp() + 1.

    # do sum
    if do_sum:
        KLD = torch.sum(KLD_element) * -0.5
        return KLD
    else:
        KLD_element = torch.sum(KLD_element, 1) * -0.5
        return KLD_element

def log_mean_exp(x):
    '''
    Performs log-mean-exp
    :param x: Tensor (batch_size, K) where K is the number of IWAE samples
    :return: Tensor (batch_size) where each entry is log-mean-exp of each row
    '''
    m = torch.max(x, dim=1)[0]
    return m + torch.log(torch.mean(torch.exp(x - m.view(-1,1).repeat(1,x.size(1))), dim=1))

def norm_exp(x):
    '''
    Performs batch norm of exp
    :param x: Tensor (batch_size, K) where K is the number of IWAE samples
    :return: Tensor (batch_size) where each entry is log-mean-exp of each row
    '''
    m = torch.max(x, dim=1)[0]
    w = torch.exp(x - m.view(-1,1).repeat(1,x.size(1)))
    w_sum = w.sum(dim=1).view(-1,1).repeat(1,w.size(1))
    return w/w_sum
