import torch
import torch.nn as nn
import torch.nn.functional as F


MIN_LOGVAR = -1.
MAX_LOGVAR = 2.

class NormalDistribution(nn.Module):
    def __init__(self, nonlinearity=None):
        super(NormalDistribution, self).__init__()
        self.nonlinearity = nonlinearity

    def clip_logvar(self, logvar):
        # clip logvar values
        if self.nonlinearity == 'hard':
            logvar = torch.max(logvar, MIN_LOGVAR*torch.ones_like(logvar))
            logvar = torch.min(logvar, MAX_LOGVAR*torch.ones_like(logvar))
        elif self.nonlinearity == 'softplus':
            logvar = F.softplus(logvar)
        elif self.nonlinearity == 'tanh':
            logvar = F.tanh(logvar)
        elif self.nonlinearity == '2tanh':
            logvar = 2.0*F.tanh(logvar)
        return logvar

    def sample_gaussian(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(self, input):
        mu = self.mean_fn(input)
        logvar = self.clip_logvar(self.logvar_fn(input))
        return mu, logvar

class NormalDistributionLinear(NormalDistribution):
    def __init__(self, input_size, output_size, nonlinearity=None):
        super(NormalDistributionLinear, self).__init__(nonlinearity=nonlinearity)

        self.input_size = input_size
        self.output_size = output_size

        # define net
        self.mean_fn = nn.Linear(input_size, output_size)
        self.logvar_fn = nn.Linear(input_size, output_size)

class NormalDistributionConv2d(NormalDistribution):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, nonlinearity=None):
        super(NormalDistributionConv2d, self).__init__(nonlinearity=nonlinearity)

        # define net
        self.mean_fn = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.logvar_fn = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)


class NormalDistributionConvTranspose2d(NormalDistribution):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, bias=True, nonlinearity=None):
        super(NormalDistributionConvTranspose2d, self).__init__(nonlinearity=nonlinearity)

        # define net
        self.mean_fn = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding, bias=bias)
        self.logvar_fn = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding, bias=bias)