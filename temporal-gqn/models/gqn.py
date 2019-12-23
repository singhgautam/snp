import random

import torch
import torch.nn as nn

from models.tools.rendering import Renderer
from models.tools.masasu.representation import Tower
from models.tools.reparam import NormalDistributionConv2d
from models.tools.representation import ContextNetwork, ConvLSTMCell, LSTMForwardEncoder

from utils import loss_kld_gaussian_vs_gaussian, loss_recon_gaussian_w_fixed_var, loss_recon_bernoulli, loss_recon_gaussian
from utils import tensorboardlogger, NormalizedAdder, NormalizedAdderList, ScaledNormalizedAdder, recursive_clone_structure, log_mean_exp


# MACROS
SAMPLE_USING_P = 0
SAMPLE_USING_Q = 1

IMAGES = 0
QUERIES = 1


class ConvDRAW(nn.Module):
    '''
    Modified ConvDraw for TGQN
    copy and modified from https://github.com/l3robot/pytorch-ConvDraw/blob/master/src/models.py
    In generator, emb_recon is discarded unlike original conv-draw
    '''

    def __init__(self,
                 im_height,
                 im_channels,
                 nc_enc,
                 nc_lstm,
                 nc_context,
                 nc_query,
                 nz,
                 n_draw_steps,
                 loss_type,
                 shared_core=False,
                 concat_latents=False,
                 ):
        super().__init__()
        self.im_height = im_height
        self.im_channels = im_channels
        self.nc_enc = nc_enc
        self.nc_lstm = nc_lstm
        self.nc_context = nc_context
        self.nc_query = nc_query
        self.nz = nz
        self.n_draw_steps = n_draw_steps
        self.z_height = im_height // 4
        self.loss_type = loss_type
        self.shared_core = shared_core
        self.concat_latents = concat_latents

        # define networks
        self.rnn_p = ConvLSTMCell(nz + nc_context, nc_lstm) if shared_core else \
            nn.ModuleList([ConvLSTMCell(nz + nc_context, nc_lstm) for _ in range(n_draw_steps)])
        self.reparam_p = NormalDistributionConv2d(nc_lstm, nz, kernel_size=5, stride=1, padding=2)

        self.rnn_q = ConvLSTMCell(nc_lstm + nc_context, nc_lstm) if shared_core else \
            nn.ModuleList([ConvLSTMCell(nc_lstm + nc_context, nc_lstm) for _ in range(n_draw_steps)])
        self.reparam_q = NormalDistributionConv2d(nc_lstm, nz, kernel_size=5, stride=1, padding=2)


    def forward(self, C_t, D_t):
        n_episodes = D_t.size(0)

        # init states
        state_p = (self.rnn_p if self.shared_core else self.rnn_p[0]).init_state(n_episodes,[self.z_height, self.z_height])
        state_q = (self.rnn_q if self.shared_core else self.rnn_q[0]).init_state(n_episodes,[self.z_height, self.z_height])
        hidden_p = state_p[0]

        z_t = []
        kl = ScaledNormalizedAdder(next(self.parameters()).new_zeros(1), (1.0/n_episodes))
        log_pq_ratio = NormalizedAdder(next(self.parameters()).new_zeros(n_episodes))
        for i in range(self.n_draw_steps):
            # select inference and generation core
            _rnn_p = self.rnn_p if self.shared_core else self.rnn_p[i]
            _rnn_q = self.rnn_q if self.shared_core else self.rnn_q[i]

            input_q = torch.cat([hidden_p, D_t], dim=1)
            hidden_q, state_q = _rnn_q(input_q, state_q)

            mean_q, logvar_q = self.reparam_q(hidden_q)
            mean_p, logvar_p = self.reparam_p(hidden_p)

            # sample z from q
            z_t_i = self.reparam_q.sample_gaussian(mean_q, logvar_q)

            # log p/q
            log_pq_ratio.append(
                loss_recon_gaussian(mean_q.view(n_episodes,-1), logvar_q.view(n_episodes,-1), z_t_i.view(n_episodes,-1), reduction="batch_sum")
                - loss_recon_gaussian(mean_p.view(n_episodes,-1), logvar_p.view(n_episodes,-1), z_t_i.view(n_episodes,-1), reduction="batch_sum")
            )

            # update prior rnn
            input_p = torch.cat([z_t_i, C_t], dim=1)
            hidden_p, state_p = _rnn_p(input_p, state_p)

            # append z to latent
            z_t += [z_t_i]
            kl.append(loss_kld_gaussian_vs_gaussian(mean_q, logvar_q, mean_p, logvar_p))

        # concat z
        z_t = torch.cat(z_t, dim=1) if self.concat_latents else z_t_i

        return {
            "z_t": z_t,
            "kl": kl.sum,
            "log_pz/qz_batchwise": log_pq_ratio.sum,
        }

    def generate(self, C_t):
        # init
        num_episodes = C_t.size(0)

        state_p = (self.rnn_p if self.shared_core else self.rnn_p[0]).init_state(num_episodes, [self.z_height, self.z_height])
        hidden_p = state_p[0]

        z_t = []
        for i in range(self.n_draw_steps):
            # select inference core
            _rnn_p = self.rnn_p if self.shared_core else self.rnn_p[i]

            # compute prior
            mean_p, logvar_p = self.reparam_p(hidden_p)

            z_t_i = self.reparam_p.sample_gaussian(mean_p, logvar_p)

            # update prior rnn
            input_p = torch.cat([z_t_i, C_t], dim=1)
            hidden_p, state_p = _rnn_p(input_p, state_p)

            z_t += [z_t_i]

        # concat z
        z_t = torch.cat(z_t, dim=1) if self.concat_latents else z_t_i

        return {
            "z_t": z_t,
        }

class Emission(nn.Module):
    def __init__(self, im_height, im_channels, nc_enc, nc_lstm, nc_condition, num_draw_steps, nc_query, z_height, loss_type):
        super().__init__()
        self.renderer = Renderer(
            im_height=im_height,
            im_channels=im_channels,
            nc_enc=nc_enc,
            nc_lstm=nc_lstm,
            nz=nc_condition,
            nc_query=nc_query,
            z_height=z_height,
            num_steps=num_draw_steps
        )
        self.num_draw_steps = num_draw_steps
        self.nc_query = nc_query
        self.z_height = z_height
        self.loss_type = loss_type

    def forward(self, SS, XY, std=1.0, pdT=None):
        condition = []
        Y = []
        X = []
        for t in range(len(XY)):
            if pdT is not None and (pdT[t] == SAMPLE_USING_P):
                continue
            for b in range(len(XY[t])):
                Y += [XY[t][b][IMAGES]]
                X += [XY[t][b][QUERIES].view(-1, self.nc_query, 1, 1).repeat(1, 1,self.z_height, self.z_height)]
                n_Y_t_b = len(XY[t][b][IMAGES])
                condition += [SS[t]["z_t"][b].unsqueeze(0).repeat(n_Y_t_b,1,1,1)]

        if len(Y) == 0:
            return {
                "nll_per_t": {t:None for t in range(len(XY))},
                "nll_per_t_per_b": {t: {b: None for b in range(len(XY[t]))} for t in range(len(XY))},
                "nll": None,
                "mu_Y": {t:{b:None for b in range(len(XY[t]))} for t in range(len(XY))},
            }

        Y = torch.cat(Y, dim=0)
        X = torch.cat(X, dim=0)
        condition = torch.cat(condition, dim=0)

        n_Y = len(Y)

        mu_Y = self.renderer(condition, X, self.num_draw_steps)

        if self.loss_type == "bernoulli":
            mu_Y = torch.sigmoid(mu_Y)
            recon_nll_imgwise = loss_recon_bernoulli(mu_Y.view(n_Y,-1), Y.view(n_Y,-1), reduction="batch_sum")
        elif self.loss_type == "scalar_gaussian":
            recon_nll_imgwise = loss_recon_gaussian_w_fixed_var(mu_Y.view(n_Y, -1), Y.view(n_Y, -1), reduction="batch_sum", std=1.0)
        else:
            raise NotImplementedError

        ret = {
            "nll_per_t": {t:None for t in range(len(XY))},
            "nll_per_t_per_b": {t: {b: None for b in range(len(XY[t]))} for t in range(len(XY))},
            "nll": torch.mean(recon_nll_imgwise),
            "mu_Y": {t:{b:None for b in range(len(XY[t]))} for t in range(len(XY))},
        }

        cursor = 0
        for t in range(len(XY)):
            if pdT is not None and (pdT[t] == SAMPLE_USING_P):
                continue
            cursor_t = cursor
            for b in range(len(XY[t])):
                n_Y_t_b = len(XY[t][b][IMAGES])
                ret["mu_Y"][t][b] = mu_Y[cursor_t:cursor_t+n_Y_t_b].detach()
                ret["nll_per_t_per_b"][t][b] = torch.mean(recon_nll_imgwise[cursor_t:cursor_t + n_Y_t_b])
                cursor_t += n_Y_t_b
            ret["nll_per_t"][t] = torch.mean(recon_nll_imgwise[cursor:cursor_t])
            cursor = cursor_t

        return ret

    def generate(self, SS, X):
        condition = []
        XX = []
        for t in range(len(X)):
            for b in range(len(X[t])):
                XX += [X[t][b].view(-1, self.nc_query, 1, 1).repeat(1, 1, self.z_height, self.z_height)]
                n_Y_t_b = len(X[t][b])
                condition += [SS[t]["z_t"][b].unsqueeze(0).repeat(n_Y_t_b, 1, 1, 1)]

        XX = torch.cat(XX, dim=0)
        condition = torch.cat(condition, dim=0)

        mu_Y = self.renderer(condition, XX, self.num_draw_steps)

        if self.loss_type == "bernoulli":
            mu_Y = torch.sigmoid(mu_Y)
        elif self.loss_type == "scalar_gaussian":
            pass
        else:
            raise NotImplementedError

        ret = {
            "mu_Y": {t:{b:None for b in range(len(X[t]))} for t in range(len(X))},
        }

        cursor = 0
        for t in range(len(X)):
            cursor_t = cursor
            for b in range(len(X[t])):
                n_Y_t_b = len(X[t][b])
                ret["mu_Y"][t][b] = mu_Y[cursor_t:cursor_t+n_Y_t_b].detach()
                cursor_t += n_Y_t_b
            cursor = cursor_t

        return ret




class GQN(nn.Module):
    def __init__(self,
                 im_height,
                 im_channels,
                 nc_enc=32,
                 nc_lstm=64,
                 nc_context=256,
                 nc_query=7,
                 nc_z=3,
                 n_draw_steps=4,
                 n_timesteps=20,
                 loss_type='bernoulli',
                 n_actions=0,
                 action_emb_size=32,
                 shared_core=False,
                 concat_latents=False,
                 ):
        super().__init__()
        self.nz = nc_z
        self.num_timesteps = n_timesteps
        self.n_draw_steps = n_draw_steps

        self.convdraw = ConvDRAW(
                im_height,
                im_channels,
                nc_enc,
                nc_lstm,
                nc_context,
                nc_query,
                nc_z,
                n_draw_steps,
                loss_type,
                shared_core=shared_core,
                concat_latents=concat_latents,
        )


        self.concat_latents = concat_latents
        self.z_height = self.convdraw.z_height
        self.repnet = ContextNetwork(repnet=Tower(nc=im_channels, nc_query=nc_query + (action_emb_size if n_actions > 0 else 1)), train_init_representation=True)
        self.n_actions = n_actions

        self.reps_nc, self.reps_nh, self.reps_nw = self.repnet.repnet.get_output_size()

        if n_actions > 0:
            self.action_encoder = LSTMForwardEncoder(input_size=n_actions,
                                                     hidden_size=action_emb_size,
                                                     train_init_state=True)

        self.emission = Emission(
            im_height=im_height,
            im_channels=im_channels,
            nc_enc=nc_enc,
            nc_lstm=nc_lstm,
            nc_condition=nc_z * (n_draw_steps if concat_latents else 1),
            num_draw_steps=n_draw_steps,
            nc_query=nc_query + (action_emb_size if n_actions > 0 else 1),
            z_height=self.reps_nh,
            loss_type=loss_type,
        )

        # attribute logging
        tensorboardlogger.log_class_attributes(self, "model/gqn/")
        tensorboardlogger.log_class_attributes(self.repnet, "model/context_network/")
        tensorboardlogger.log_class_attributes(self.convdraw, "model/convdraw/")

    def forward(self, C, XY, A=None, params = {}):

        # parse params
        beta = params["beta"] if "beta" in params else 1.0
        std = params["std"] if "std" in params else 1.0

        # init prelims
        n_episodes = len(C[0])
        n_timesteps = len(C)

        # loss adders
        loss_nll_adder = NormalizedAdder(next(self.parameters()).new_zeros(1))
        loss_kl_adder = NormalizedAdder(next(self.parameters()).new_zeros(1))
        info_kl_t_adder = NormalizedAdderList(next(self.parameters()).new_zeros(1), n_timesteps)

        # init reps
        AA = []

        # expand actions
        for t in range(n_timesteps):
            if self.n_actions > 0:
                a_t = torch.cat([a_t_b.unsqueeze(0) for a_t_b in A[t]], dim=0)
            else:
                a_t = next(self.parameters()).new_ones((n_episodes,1))*(t/50.) # hardcode 50. to have same scale for different data sets
            AA += [a_t]
        if self.n_actions > 0:
            AA = self.action_encoder(AA)

        # append action to C
        CA = recursive_clone_structure(C)
        for t in range(len(CA)):
            for b in range(len(CA[t])):
                if CA[t][b][QUERIES] is not None:
                    n_X_t_b = len(CA[t][b][QUERIES])
                    CA[t][b][QUERIES] = torch.cat([CA[t][b][QUERIES], AA[t][b].unsqueeze(0).repeat(n_X_t_b, 1)], dim=1)

        # append action to X
        XYA = recursive_clone_structure(XY)
        for t in range(len(XYA)):
            for b in range(len(XYA[t])):
                n_X_t_b = len(XYA[t][b][QUERIES])
                XYA[t][b][QUERIES] = torch.cat([XYA[t][b][QUERIES], AA[t][b].unsqueeze(0).repeat(n_X_t_b, 1)], dim=1)

        # scramble XYA for reconstruction
        XYA_scramble = recursive_clone_structure(XYA)
        XYA_QUERIES = [[] for _ in range(n_episodes)]
        XYA_IMAGES = [[] for _ in range(n_episodes)]
        for b in range(n_episodes):
            # collate all time-steps
            XYA_QUERIES[b] = torch.cat([XYA[t][b][QUERIES] for t in range(n_timesteps)], dim=0)
            XYA_IMAGES[b] = torch.cat([XYA[t][b][IMAGES] for t in range(n_timesteps)], dim=0)

            # perform scrambling
            for t in range(n_timesteps):
                indices = random.sample(range(len(XYA_QUERIES[b])), len(XYA_scramble[t][b]))
                XYA_scramble[t][b][QUERIES] = XYA_QUERIES[b][indices]
                XYA_scramble[t][b][IMAGES] = XYA_IMAGES[b][indices]

        # compute representations
        R_CA = []
        R_CXYA = 0
        for t in range(n_timesteps):
            # compute context representation and record cumulative context
            R_CA_t = None
            if CA[t][b][IMAGES] is not None and CA[t][b][QUERIES] is not None:
                R_CA_t = self.repnet(CA[t])

            if R_CA_t is not None:
                R_CA = R_CA + [R_CA[t-1] + R_CA_t] if len(R_CA) > 0 else [R_CA_t]
            else:
                R_CA = R_CA + [R_CA[t-1]]

            # compute target representation
            R_XYA_t = None
            if XYA[t][b][IMAGES] is not None and XYA[t][b][QUERIES] is not None:
                R_XYA_t = self.repnet(XYA[t])

            # add context and target for inference
            if R_CA_t is not None:
                R_CXYA += R_CA_t
            if R_XYA_t is not None:
                R_CXYA += R_XYA_t

        SS = {
            "gqn": {},
        }

        for t in range(n_timesteps):
            response = self.convdraw(R_CA[t], R_CXYA)
            loss_kl_adder.append(response["kl"])
            info_kl_t_adder[t].append(response["kl"])
            SS["gqn"][t] = {'z_t': response["z_t"]}

        # emission visual
        emission_gqn = self.emission(SS["gqn"], XYA, std=std)
        mu_Y = {"gqn": emission_gqn["mu_Y"]}

        # emission reconstruction
        emission_gqn = self.emission(SS["gqn"], XYA_scramble, std=std)
        loss_nll_adder.append(emission_gqn["nll"])

        ret = {
            "info_scalar":{
                "kl" : loss_kl_adder.mean().detach().item(),
                "recon_nll" : loss_nll_adder.mean().detach().item(),
                "elbo": loss_kl_adder.mean().detach().item() + loss_nll_adder.mean().detach().item(),
                "loss": beta*loss_kl_adder.mean().detach().item() + loss_nll_adder.mean().detach().item(),
            },
            "info_temporal":{
                "kl_t_gqn": [item.detach().item() for item in info_kl_t_adder.mean_list()],
                "recon_nll_t_gqn": [emission_gqn["nll_per_t"][t].detach().item() for t in emission_gqn["nll_per_t"]],
            },
            "__loss__": beta * loss_kl_adder.mean() + loss_nll_adder.mean(),
            "__mu_Y__" : mu_Y,
        }

        return ret

    def generate(self, C, X, A=None):
        # init prelims
        n_episodes = len(C[0])
        n_timesteps = len(C)

        # init reps
        AA = []

        # expand actions
        for t in range(n_timesteps):
            if self.n_actions > 0:
                a_t = torch.cat([a_t_b.unsqueeze(0) for a_t_b in A[t]], dim=0)
            else:
                a_t = next(self.parameters()).new_ones((n_episodes, 1)) * (t / 50.)  # hardcode 50. to have same scale for different data sets
            AA += [a_t]
        if self.n_actions > 0:
            AA = self.action_encoder(AA)

        # append action to C
        CA = recursive_clone_structure(C)
        for t in range(len(CA)):
            for b in range(len(CA[t])):
                if CA[t][b][QUERIES] is not None:
                    n_X_t_b = len(CA[t][b][QUERIES])
                    CA[t][b][QUERIES] = torch.cat([CA[t][b][QUERIES], AA[t][b].unsqueeze(0).repeat(n_X_t_b, 1)], dim=1)

        # append action to X
        XA = recursive_clone_structure(X)
        for t in range(len(XA)):
            for b in range(len(XA[t])):
                n_X_t_b = len(XA[t][b])
                XA[t][b] = torch.cat([XA[t][b], AA[t][b].unsqueeze(0).repeat(n_X_t_b, 1)], dim=1)

        # compute representations
        R_CA = []
        for t in range(n_timesteps):
            # compute context representation and record cumulative context
            R_CA_t = None
            if CA[t][b][IMAGES] is not None and CA[t][b][QUERIES] is not None:
                R_CA_t = self.repnet(CA[t])

            if R_CA_t is not None:
                R_CA = R_CA + [R_CA[t-1] + R_CA_t] if len(R_CA) > 0 else [R_CA_t]
            else:
                R_CA = R_CA + [R_CA[t-1]]

        SS = {
            "gqn": {},
        }

        for t in range(n_timesteps):
            response = self.convdraw.generate(R_CA[t])
            SS["gqn"][t] = {'z_t': response["z_t"]}

        # emission
        emission_gqn = self.emission.generate(SS["gqn"], XA)
        mu_Y = {"gqn": emission_gqn["mu_Y"]}

        ret = {
            "__mu_Y__": mu_Y,
        }

        return ret

    def evaluate(self, C, XY, A=None, params = {}):

        # parse params
        beta = params["beta"] if "beta" in params else 1.0
        std = params["std"] if "std" in params else 1.0
        K = params["K"] if "K" in params else 50
        eval_nll = params["eval_nll"] if "eval_nll" in params else True
        eval_mse = params["eval_mse"] if "eval_mse" in params else True

        # init prelims
        n_episodes = len(C[0])
        n_timesteps = len(C)

        # loss adders
        info_kl_adder = NormalizedAdder(next(self.parameters()).new_zeros(1))
        info_kl_t_adder = NormalizedAdderList(next(self.parameters()).new_zeros(1), n_timesteps)

        info_recon_nll_adder = NormalizedAdder(next(self.parameters()).new_zeros(1))
        info_recon_nll_t_adder = NormalizedAdderList(next(self.parameters()).new_zeros(1), n_timesteps)

        info_gen_nll_adder = NormalizedAdder(next(self.parameters()).new_zeros(1))
        info_gen_nll_t_adder = NormalizedAdderList(next(self.parameters()).new_zeros(1), n_timesteps)

        iwae_info = {
            "log_p_y_giv_xz__per_t" : next(self.parameters()).new_zeros((n_timesteps, n_episodes, K)),
            "log_pz/qz__per_t": next(self.parameters()).new_zeros((n_timesteps, n_episodes, K)),
        }

        # init reps
        AA = []

        # expand actions
        for t in range(n_timesteps):
            if self.n_actions > 0:
                a_t = torch.cat([a_t_b.unsqueeze(0) for a_t_b in A[t]], dim=0)
            else:
                a_t = next(self.parameters()).new_ones((n_episodes,1))*(t/50.) # hardcode 50. to have same scale for different data sets
            AA += [a_t]
        if self.n_actions > 0:
            AA = self.action_encoder(AA)

        # append action to C
        CA = recursive_clone_structure(C)
        for t in range(len(CA)):
            for b in range(len(CA[t])):
                if CA[t][b][QUERIES] is not None:
                    n_X_t_b = len(CA[t][b][QUERIES])
                    CA[t][b][QUERIES] = torch.cat([CA[t][b][QUERIES], AA[t][b].unsqueeze(0).repeat(n_X_t_b, 1)], dim=1)

        # append action to X
        XYA = recursive_clone_structure(XY)
        for t in range(len(XYA)):
            for b in range(len(XYA[t])):
                n_X_t_b = len(XYA[t][b][QUERIES])
                XYA[t][b][QUERIES] = torch.cat([XYA[t][b][QUERIES], AA[t][b].unsqueeze(0).repeat(n_X_t_b, 1)], dim=1)

        # compute representations
        R_CA = []
        R_CXYA = 0
        for t in range(n_timesteps):
            # compute context representation and record cumulative context
            R_CA_t = None
            if CA[t][b][IMAGES] is not None and CA[t][b][QUERIES] is not None:
                R_CA_t = self.repnet(CA[t])

            if R_CA_t is not None:
                R_CA = R_CA + [R_CA[t - 1] + R_CA_t] if len(R_CA) > 0 else [R_CA_t]
            else:
                R_CA = R_CA + [R_CA[t - 1]]

            # compute target representation
            R_XYA_t = None
            if XYA[t][b][IMAGES] is not None and XYA[t][b][QUERIES] is not None:
                R_XYA_t = self.repnet(XYA[t])

            # add context and target for inference
            if R_CA_t is not None:
                R_CXYA += R_CA_t
            if R_XYA_t is not None:
                R_CXYA += R_XYA_t

        if eval_nll:
            for k in range(K):
                SS = {
                    "gqn": {},
                }
                for t in range(n_timesteps):
                    response = self.convdraw(R_CA[t], R_CXYA)

                    # record kl
                    info_kl_adder.append(response["kl"].detach())
                    info_kl_t_adder[t].append(response["kl"].detach())

                    # record log p/q
                    iwae_info["log_pz/qz__per_t"][t, :, k] = response["log_pz/qz_batchwise"].detach()

                    # record states
                    SS["gqn"][t] = {'z_t': response["z_t"]}

                # emission
                emission_gqn = self.emission(SS["gqn"], XYA, std=std)

                # record recon nll
                info_recon_nll_adder.append(emission_gqn["nll"].detach())
                info_recon_nll_t_adder.append_list([emission_gqn["nll_per_t"][t].detach() for t in emission_gqn["nll_per_t"]])

                # record log p_y_giv_xz
                for t in emission_gqn["nll_per_t_per_b"]:
                    for b in emission_gqn["nll_per_t_per_b"][t]:
                        iwae_info["log_p_y_giv_xz__per_t"][t, b, k] = -emission_gqn["nll_per_t_per_b"][t][b].detach()

        # basic metrics
        ret = {
            "info_scalar":{
                "kl" : info_kl_adder.mean().detach().item(),
                "recon_nll" : info_recon_nll_adder.mean().detach().item(),
                "elbo": info_kl_adder.mean().detach().item() + info_recon_nll_adder.mean().detach().item(),
            },
            "info_temporal":{
                "kl_t": [item.detach().item() for item in info_kl_t_adder.mean_list()],
                "recon_nll_t": [item.detach().item() for item in info_recon_nll_t_adder.mean_list()],
            },
        }

        # elbo per timestep
        ret["info_temporal"]["elbo_t"] = [(ret["info_temporal"]["kl_t"][t] + ret["info_temporal"]["recon_nll_t"][t]) for t in range(n_timesteps)]

        # iwae nll
        ret["info_scalar"]["iwae_nll"] = -torch.mean(log_mean_exp(torch.sum(iwae_info["log_p_y_giv_xz__per_t"], dim=0) + torch.sum(iwae_info["log_pz/qz__per_t"], dim=0))).detach().item()
        ret["info_temporal"]["iwae_nll_t"] = [
            -torch.mean(log_mean_exp(iwae_info["log_p_y_giv_xz__per_t"][t] + iwae_info["log_pz/qz__per_t"][t])).detach().item()
        for t in range(n_timesteps)]

        # gen nll
        if eval_mse:
            for k in range(K):
                SS = {
                    "gqn": {},
                }

                for t in range(n_timesteps):
                    response = self.convdraw.generate(R_CA[t])
                    SS["gqn"][t] = {'z_t': response["z_t"]}

                # emission
                emission_gqn = self.emission(SS["gqn"], XYA, std=std)

                # record nll
                info_gen_nll_adder.append(emission_gqn["nll"].detach())
                info_gen_nll_t_adder.append_list([emission_gqn["nll_per_t"][t].detach() for t in emission_gqn["nll_per_t"]])

        ret["info_scalar"]["gen_nll"] = info_gen_nll_adder.mean().detach().item()
        ret["info_temporal"]["gen_nll_t"] = [item.detach().item() for item in info_gen_nll_t_adder.mean_list()]

        return ret