'''
copy and modified from https://github.com/l3robot/pytorch-ConvDraw/blob/master/src/models.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.tools.representation import  ConvLSTMCell

class RendererDecoder(nn.Module):
    '''
    copy and modified from https://github.com/l3robot/pytorch-ConvDraw/blob/master/src/models.py
    '''
    def __init__(self, im_channels, h_dim, lstm_dim):
        super().__init__()
        self.decode = nn.Conv2d(lstm_dim, h_dim, 5, stride=1, padding=2)
        self.convt = nn.ConvTranspose2d(h_dim, h_dim*2, 4, stride=2, padding=1)
        self.convt2 = nn.ConvTranspose2d(h_dim*2, im_channels, 4, stride=2, padding=1)
        #self.reparam = NormalDistributionConvTranspose2d(h_dim*2, im_channels, kernel_size=4, stride=2, padding=1)

    def sample(self, mu, logvar):
        return self.reparam.sample_gaussian(mu, logvar)

    def forward(self, h):
        xx = F.relu(self.decode(h))
        xx = F.relu(self.convt(xx))
        #mu, logvar = self.reparam(xx)
        #return mu, logvar
        return self.convt2(xx)

class RendererEncoder(nn.Module):
    '''
    copy and modified from https://github.com/l3robot/pytorch-ConvDraw/blob/master/src/models.py
    '''
    def __init__(self, im_channels, h_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(im_channels, h_dim*2, 4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(h_dim*2, h_dim, 4, stride=2, padding=1)

    def forward(self, x):
        hidden = F.relu(self.conv1(x))
        return F.relu(self.conv2(hidden))

class AttentionKeyDecoder(nn.Module):
    def __init__(self, h_dim, key_size):
        super().__init__()
        self.conv1 = nn.Conv2d(h_dim, h_dim, 1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(h_dim, key_size, 1, stride=1, padding=0)
        self.pool = nn.AvgPool2d(16)

    def forward(self, x):
        hidden = F.relu(self.conv1(x))
        hidden = F.relu(self.conv2(hidden))
        return self.pool(hidden)

class AttentionValueDecoder(nn.Module):
    def __init__(self, h_dim, out_size):
        super().__init__()
        self.conv1 = nn.Conv2d(h_dim, h_dim, 1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(h_dim, out_size, 1, stride=1, padding=0)

    def forward(self, x):
        hidden = F.relu(self.conv1(x))
        hidden = F.relu(self.conv2(hidden))
        return hidden



class Renderer(nn.Module):
    '''
    Modified ConvDraw for GQN
    copy and modified from https://github.com/l3robot/pytorch-ConvDraw/blob/master/src/models.py
    in generator, emb_recon is discarded unlike original conv-draw
    '''
    def __init__(self,
                 im_height=64,  # image height
                 im_channels=3,  # number of channels in image
                 nc_enc=128,  # kernel size (number of channels) for encoder
                 nc_lstm=64,  # kernel size (number of channels) for lstm
                 nc_query=7,  # kernel size (number of channels) for query
                 nz=3*4,  # size of latent variable
                 z_height=8,  # height of feature map size for z
                 num_steps=4,  # number of steps in Renderer (no need to be the same size as Draw's num_steps)
                 use_canvas_in_prior=False,  # canvas usage in prior (gqn paper didn't use canvas)
                 ):
        super().__init__()
        self.im_height = im_height
        self.im_channels = im_channels
        self.nc_enc = nc_enc
        self.nc_lstm = nc_lstm
        self.nz = nz
        self.num_steps = num_steps
        self.z_height = z_height
        self.use_canvas_in_prior = use_canvas_in_prior

        # define networks
        if self.use_canvas_in_prior:
            self.encoder = RendererEncoder(im_channels, nc_enc)
            self.rnn_p = ConvLSTMCell(nc_enc + nz + nc_query, nc_lstm)
        else:
            self.rnn_p = ConvLSTMCell(nz + nc_query, nc_lstm)
        self.decoder = RendererDecoder(im_channels, nc_enc, nc_lstm)


    def forward(self, z, emb_query, num_steps=None):
        # init
        num_data = z.size(0)

        # init recon image
        mean_recon = z.new_zeros(num_data, self.im_channels, self.im_height, self.im_height)

        # init states
        state_p = self.rnn_p.init_state(num_data, [self.z_height, self.z_height])
        hidden_p = state_p[0]
        for i in range(num_steps if num_steps is not None else self.num_steps):
            # update rnn
            if self.use_canvas_in_prior:
                emb_recon = self.encoder(mean_recon)
                input_p = torch.cat([z, emb_recon, emb_query], dim=1)
            else:
                input_p = torch.cat([z, emb_query], dim=1)
            hidden_p, state_p = self.rnn_p(input_p, state_p)

            # update recon
            dmean_recon = self.decoder(hidden_p)
            mean_recon = mean_recon + dmean_recon

        # return
        return mean_recon