import numpy as np

import torch
import torch.nn as nn

from models.tools.rendering import Renderer
from models.tools.masasu.representation import Tower
from models.tools.reparam import NormalDistributionConv2d
from models.tools.representation import ContextNetwork, ConvLSTMCell, BiConvLSTM, ConvLSTMBackwardEncoder

from utils import loss_kld_gaussian_vs_gaussian, loss_recon_gaussian_w_fixed_var, loss_recon_bernoulli, loss_recon_gaussian, log_mean_exp, norm_exp, stratified_resampling
from utils import tensorboardlogger, NormalizedAdder, NormalizedAdderList, ScaledNormalizedAdder, recursive_detach


# MACROS
SAMPLE_USING_P = 0
SAMPLE_USING_Q = 1

IMAGES = 0
QUERIES = 1


class TConvDRAW(nn.Module):
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
                 n_timesteps,
                 nc_ssm,
                 loss_type,
                 n_actions,
                 shared_core=False,
                 concat_latents=False,
                 use_ssm_context=False,
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
        self.n_timesteps = n_timesteps
        self.nc_ssm = nc_ssm
        self.loss_type = loss_type
        self.n_actions = n_actions
        self.shared_core = shared_core
        self.concat_latents = concat_latents
        self.use_ssm_context = use_ssm_context

        # define networks
        self.rnn_p = ConvLSTMCell(nz + (nc_context if use_ssm_context else 0) + nc_ssm + n_actions, nc_lstm) if shared_core else \
            nn.ModuleList([ConvLSTMCell(nz + nc_context + nc_ssm + n_actions, nc_lstm) for _ in range(n_draw_steps)])
        self.reparam_p = NormalDistributionConv2d(nc_lstm, nz, kernel_size=5, stride=1, padding=2)

        self.rnn_q = ConvLSTMCell(nc_lstm + nc_context + nc_ssm + n_actions, nc_lstm) if shared_core else \
            nn.ModuleList([ConvLSTMCell(nc_lstm + nc_context + nc_ssm + n_actions, nc_lstm) for _ in range(n_draw_steps)])
        self.reparam_q = NormalDistributionConv2d(nc_lstm, nz, kernel_size=5, stride=1, padding=2)

        # sSSM transition state
        self.ssm = ConvLSTMCell(
            input_size=nz * (n_draw_steps if concat_latents else 1) + (nc_context if use_ssm_context else 0) + n_actions,
            hidden_size=self.nc_ssm, train_init_state=True, height=self.z_height, width=self.z_height,
        )

    def forward(self, C_t, D_t, a_tmo, cs_tmo, z_tmo):
        n_episodes = D_t.size(0)
        s_tmo, _ = cs_tmo

        # SSM transition
        s_t, cs_t = self.ssm(torch.cat([z_tmo, C_t, a_tmo] if self.use_ssm_context else [z_tmo, a_tmo], dim=1), cs_tmo)

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

            input_q = torch.cat([hidden_p, D_t, s_t, a_tmo], dim=1)
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
            input_p = torch.cat([z_t_i, C_t, s_t, a_tmo], dim=1)
            hidden_p, state_p = _rnn_p(input_p, state_p)

            # append z to latent
            z_t += [z_t_i]
            kl.append(loss_kld_gaussian_vs_gaussian(mean_q, logvar_q, mean_p, logvar_p))

        # concat z
        z_t = torch.cat(z_t, dim=1) if self.concat_latents else z_t_i

        return {
            "z_t": z_t,
            "cs_t": cs_t,
            "kl": kl.sum,
            "log_pz/qz_batchwise": log_pq_ratio.sum,
        }

    def generate(self, C_t, a_tmo, cs_tmo, z_tmo):
        # init
        n_episodes = C_t.size(0)

        # SSM
        s_tmo, _ = cs_tmo
        s_t, cs_t = self.ssm(torch.cat([z_tmo, C_t, a_tmo] if self.use_ssm_context else [z_tmo, a_tmo], dim=1), cs_tmo)

        state_p = (self.rnn_p if self.shared_core else self.rnn_p[0]).init_state(n_episodes, [self.z_height, self.z_height])
        hidden_p = state_p[0]

        z_t = []
        for i in range(self.n_draw_steps):
            # select inference core
            _rnn_p = self.rnn_p if self.shared_core else self.rnn_p[i]

            # compute prior
            mean_p, logvar_p = self.reparam_p(hidden_p)

            z_t_i = self.reparam_p.sample_gaussian(mean_p, logvar_p)

            # update prior rnn
            input_p = torch.cat([z_t_i, C_t, s_t, a_tmo], dim=1)
            hidden_p, state_p = _rnn_p(input_p, state_p)

            z_t += [z_t_i]

        # concat z
        z_t = torch.cat(z_t, dim=1) if self.concat_latents else z_t_i

        return {
            "z_t": z_t,
            "cs_t": cs_t,
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
                condition += [torch.cat([SS[t]["cs_t"][0][b], SS[t]["z_t"][b]], dim=0).unsqueeze(0).repeat(n_Y_t_b,1,1,1)]

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
            "nll_per_t_per_b": {t:{b:None for b in range(len(XY[t]))} for t in range(len(XY))},
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
                ret["nll_per_t_per_b"][t][b] = torch.mean(recon_nll_imgwise[cursor_t:cursor_t+n_Y_t_b])
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
                condition += [torch.cat([SS[t]["cs_t"][0][b], SS[t]["z_t"][b]], dim=0).unsqueeze(0).repeat(n_Y_t_b, 1, 1, 1)]

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




class TGQN_PD(nn.Module):
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
                 nc_ssm=3 * 4,
                 loss_type='bernoulli',
                 context_type='same',
                 n_actions=0,
                 shared_core=False,
                 concat_latents=False,
                 use_ssm_context=False,
                 q_bernoulli=None,
                 pd=True,
                 ):
        super().__init__()
        self.nz = nc_z
        self.num_timesteps = n_timesteps
        self.n_draw_steps = n_draw_steps

        self.tconvdraw = TConvDRAW(
                im_height, im_channels,
                nc_enc, nc_lstm, nc_context, nc_query,
                nc_z,
                n_draw_steps,
                n_timesteps,
                nc_ssm,
                loss_type=loss_type,
                n_actions=n_actions,
                shared_core=shared_core,
                concat_latents=concat_latents,
                use_ssm_context=use_ssm_context,
        )


        self.concat_latents = concat_latents
        self.z_height = self.tconvdraw.z_height
        self.repnet = ContextNetwork(repnet=Tower(nc=im_channels, nc_query=nc_query), train_init_representation=True)
        self.context_type = context_type
        self.num_actions = n_actions
        self.pd = pd

        self.q_bernoulli = [0.5]*(self.num_timesteps+1) if q_bernoulli is None else q_bernoulli

        self.init_latent_param = torch.nn.Parameter(torch.randn((1, self.nz*self.n_draw_steps if self.concat_latents else self.nz, self.z_height, self.z_height)))

        self.reps_nc, self.reps_nh, self.reps_nw = self.repnet.repnet.get_output_size()

        if context_type in ['all']:
            self.inference_encoder = BiConvLSTM(input_size=self.reps_nc + n_actions,
                                                hidden_size=self.reps_nc,
                                                hidden_height=self.reps_nh,
                                                hidden_width=self.reps_nw)
        if context_type in ['backward']:
            self.inference_encoder = ConvLSTMBackwardEncoder(input_size=self.reps_nc + n_actions,
                                                             hidden_size=self.reps_nc,
                                                             hidden_height=self.reps_nh,
                                                             hidden_width=self.reps_nw)

        self.emission = Emission(
            im_height=im_height,
            im_channels=im_channels,
            nc_enc=nc_enc,
            nc_lstm=nc_lstm,
            nc_condition=nc_z * (n_draw_steps if concat_latents else 1) + nc_ssm,
            num_draw_steps=n_draw_steps,
            nc_query=nc_query,
            z_height=self.reps_nh,
            loss_type=loss_type,
        )

        # attribute logging
        tensorboardlogger.log_class_attributes(self, "model/tgqn/")
        tensorboardlogger.log_class_attributes(self.repnet, "model/context_network/")
        tensorboardlogger.log_class_attributes(self.tconvdraw, "model/tconvdraw/")
        tensorboardlogger.log_class_attributes(self.inference_encoder, "model/inference_encoder/")

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
        info_kl_snp_adder = NormalizedAdder(next(self.parameters()).new_zeros(1))
        info_kl_pd_adder = NormalizedAdder(next(self.parameters()).new_zeros(1))
        info_kl_snp_t_adder = NormalizedAdderList(next(self.parameters()).new_zeros(1), n_timesteps)
        info_recon_nll_snp_adder = NormalizedAdder(next(self.parameters()).new_zeros(1))
        info_recon_nll_pd_adder = NormalizedAdder(next(self.parameters()).new_zeros(1))

        # init reps
        R_C = []
        R_XY = []
        R_CXYA = []
        AA = []

        # null embeddings
        null_R_t = self.repnet.get_init_representation(n_episodes)

        # expand actions
        for t in range(n_timesteps):
            a_t = next(self.parameters()).new_zeros((0))
            if A is not None:
                a_t = torch.cat([a_t_b.view(-1,1,1).repeat(1,self.reps_nh, self.reps_nw).unsqueeze(0) for a_t_b in A[t]], dim=0)
            AA += [a_t]

        # compute representations
        for t in range(n_timesteps):
            R_C_t = self.repnet(C[t])
            R_C += [R_C_t]

            R_XY_t = self.repnet(XY[t])
            R_XY += [R_XY_t]

            R_CXY_t = R_C_t + R_XY_t
            R_CXYA += [torch.cat([R_CXY_t, AA[t]], dim=1) if self.context_type in ['all', 'backward'] else R_CXY_t]

        if self.context_type in ['all', 'backward']:
            R_CXYA = self.inference_encoder(R_CXYA)
        elif self.context_type in ['same']:
            R_CXYA = R_CXYA
        else:
            raise NotImplementedError("Not implemented context_type {}".format(self.context_type))
        R_CXYA += [null_R_t]

        # initialize hidden states and latents
        cs_t = self.tconvdraw.ssm.init_state(n_episodes, [self.z_height, self.z_height])
        z_t = self.init_latent(n_episodes)
        SS = {
            "snp": {-1: {'cs_t': cs_t, 'z_t': z_t}},
            "pd": {-1: {'cs_t': cs_t, 'z_t': z_t}},
        }

        qp_choices = np.random.binomial(1, self.q_bernoulli, len(self.q_bernoulli))
        for t in range(n_timesteps):
            if self.pd:
                # PD Rollout
                choice = qp_choices[t]
                if choice == SAMPLE_USING_Q:
                    response = self.tconvdraw(R_C[t], R_CXYA[t], AA[t], SS["pd"][t-1]["cs_t"], SS["pd"][t-1]["z_t"])
                    loss_kl_adder.append(response["kl"])
                    info_kl_pd_adder.append(response["kl"])
                    SS["pd"][t] = {'cs_t': response["cs_t"], 'z_t': response["z_t"]}
                elif choice == SAMPLE_USING_P:
                    response = self.tconvdraw.generate(R_C[t], AA[t], SS["pd"][t-1]["cs_t"], SS["pd"][t-1]["z_t"])
                    SS["pd"][t] = {'cs_t': response["cs_t"], 'z_t': response["z_t"]}

            # Regular Rollout
            response = self.tconvdraw(R_C[t], R_CXYA[t], AA[t], SS["snp"][t - 1]["cs_t"], SS["snp"][t - 1]["z_t"])
            loss_kl_adder.append(response["kl"])
            info_kl_snp_adder.append(response["kl"])
            info_kl_snp_t_adder[t].append(response["kl"])
            SS["snp"][t] = {'cs_t': response["cs_t"], 'z_t': response["z_t"]}

        # emission
        mu_Y = {}

        # snp emission
        emission_snp = self.emission(SS["snp"], XY, std=std)
        loss_nll_adder.append(emission_snp["nll"])
        info_recon_nll_snp_adder.append(emission_snp["nll"])
        mu_Y["snp"] = emission_snp["mu_Y"]

        # pd emission
        if self.pd:
            emission_pd = self.emission(SS["pd"], XY, std=std, pdT=qp_choices)
            loss_nll_adder.append(emission_pd["nll"])
            info_recon_nll_pd_adder.append(emission_pd["nll"])
            mu_Y["pd"] = emission_pd["mu_Y"]

        ret = {
            "info_scalar":{
                "kl" : loss_kl_adder.mean().detach().item(),
                "kl_snp" : info_kl_snp_adder.mean().detach().item(),
                "kl_pd" : info_kl_pd_adder.mean().detach().item(),
                "recon_nll" : loss_nll_adder.mean().detach().item(),
                "recon_nll_snp" : info_recon_nll_snp_adder.mean().detach().item(),
                "recon_nll_pd" : info_recon_nll_pd_adder.mean().detach().item(),
                "loss": beta*loss_kl_adder.mean().detach().item() + loss_nll_adder.mean().detach().item(),
                "elbo_snp": info_kl_snp_adder.mean().detach().item() + info_recon_nll_snp_adder.mean().detach().item(),
                "elbo": info_kl_snp_adder.mean().detach().item() + info_recon_nll_snp_adder.mean().detach().item(),
            },
            "info_temporal":{
                "kl_t_snp": [item.detach().item() for item in info_kl_snp_t_adder.mean_list()],
                "recon_nll_t_snp": [emission_snp["nll_per_t"][t].detach().item() for t in emission_snp["nll_per_t"]],
            },
            "__loss__": beta * loss_kl_adder.mean() + loss_nll_adder.mean(),
            "__mu_Y__" : mu_Y,
        }

        return ret

    def init_latent(self, num_episodes):
        return self.init_latent_param.expand((num_episodes, self.nz*self.n_draw_steps if self.concat_latents else self.nz, self.z_height, self.z_height))

    def generate(self, C, X, A=None):
        # init prelims
        n_episodes = len(C[0])
        n_timesteps = len(C)

        # init reps
        R_C = []
        AA = []

        # expand actions
        for t in range(n_timesteps):
            a_t = next(self.parameters()).new_zeros((0))
            if A is not None:
                a_t = torch.cat([a_t_b.view(-1,1,1).repeat(1,self.reps_nh, self.reps_nw).unsqueeze(0) for a_t_b in A[t]], dim=0)
            AA += [a_t]

        # compute representations
        for t in range(n_timesteps):
            R_C_t = self.repnet(C[t])
            R_C += [R_C_t]

        # initialize hidden states and latents
        cs_t = self.tconvdraw.ssm.init_state(n_episodes, [self.z_height, self.z_height])
        z_t = self.init_latent(n_episodes)
        SS = {
            "snp": {-1: {"cs_t": cs_t, 'z_t': z_t}},
        }

        for t in range(n_timesteps):
            response = self.tconvdraw.generate(R_C[t], AA[t], SS["snp"][t - 1]["cs_t"], SS["snp"][t - 1]["z_t"])
            SS["snp"][t] = {"cs_t": response["cs_t"], 'z_t': response["z_t"]}

        emission_snp = self.emission.generate(SS["snp"], X)

        mu_Y = {
            "snp" : emission_snp["mu_Y"],
        }

        ret = {
            "__mu_Y__" : mu_Y
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

        smc_info = {
            "log_p_y_giv_xz__per_t" : next(self.parameters()).new_zeros((n_timesteps, n_episodes, K)),
            "log_mean_w__per_t": next(self.parameters()).new_zeros((n_timesteps, n_episodes)),
            "log_w__per_t_per_k": next(self.parameters()).new_zeros((n_timesteps, n_episodes, K)),
        }

        # init reps
        R_C = []
        R_XY = []
        R_CXYA = []
        AA = []

        # null embeddings
        null_R_t = self.repnet.get_init_representation(n_episodes)

        # expand actions
        for t in range(n_timesteps):
            a_t = next(self.parameters()).new_zeros((0))
            if A is not None:
                a_t = torch.cat([a_t_b.view(-1,1,1).repeat(1,self.reps_nh, self.reps_nw).unsqueeze(0) for a_t_b in A[t]], dim=0)
            AA += [a_t]

        # compute representations
        for t in range(n_timesteps):
            R_C_t = self.repnet(C[t])
            R_C += [R_C_t]

            R_XY_t = self.repnet(XY[t])
            R_XY += [R_XY_t]

            R_CXY_t = R_C_t + R_XY_t
            R_CXYA += [torch.cat([R_CXY_t, AA[t]], dim=1) if self.context_type in ['all', 'backward'] else R_CXY_t]

        if self.context_type in ['all', 'backward']:
            R_CXYA = self.inference_encoder(R_CXYA)
        elif self.context_type in ['same']:
            R_CXYA = R_CXYA
        else:
            raise NotImplementedError("Not implemented context_type {}".format(self.context_type))
        R_CXYA += [null_R_t]

        # initialize hidden states and latents
        cs_t0 = self.tconvdraw.ssm.init_state(n_episodes, [self.z_height, self.z_height])
        z_t0 = self.init_latent(n_episodes)

        if eval_nll:
            SS = {
                "smc": {-1: {'cs_t': [cs_t0 for _ in range(K)], 'z_t': [z_t0 for _ in range(K)]}},
            }

            for t in range(n_timesteps):
                log_W = next(self.parameters()).new_zeros((n_episodes, K))
                SS["smc"][t] = {'cs_t': [None for _ in range(K)], 'z_t': [None for _ in range(K)]}
                for k in range(K):
                    response = self.tconvdraw(R_C[t], R_CXYA[t], AA[t], SS["smc"][t - 1]["cs_t"][k], SS["smc"][t - 1]["z_t"][k])

                    log_W[:,k] = response["log_pz/qz_batchwise"].detach()

                    SS["smc"][t]["cs_t"][k] = recursive_detach(response["cs_t"])
                    SS["smc"][t]["z_t"][k] = recursive_detach(response["z_t"])

                    # record kl
                    info_kl_adder.append(response["kl"].detach())
                    info_kl_t_adder[t].append(response["kl"].detach())

                # resampling
                smc_info["log_w__per_t_per_k"][t] = log_W
                smc_info["log_mean_w__per_t"][t] = log_mean_exp(log_W)
                ancestors = stratified_resampling(
                    norm_exp(log_W)
                )
                for b in range(n_episodes):
                    for i, k in enumerate(ancestors[b]):
                        SS["smc"][t]["cs_t"][i][0][b] = SS["smc"][t]["cs_t"][k][0][b]
                        SS["smc"][t]["cs_t"][i][1][b] = SS["smc"][t]["cs_t"][k][1][b]
                        SS["smc"][t]["z_t"][i][b] = SS["smc"][t]["z_t"][k][b]

            for k in range(K):
                emission_smc = self.emission({t:{'cs_t': SS["smc"][t]['cs_t'][k], 'z_t': SS["smc"][t]['z_t'][k]} for t in SS["smc"]}, XY, std=std)

                # record recon nll
                info_recon_nll_adder.append(emission_smc["nll"].detach())
                info_recon_nll_t_adder.append_list([emission_smc["nll_per_t"][t].detach() for t in emission_smc["nll_per_t"]])

                # record log p_y_giv_xz
                for t in emission_smc["nll_per_t_per_b"]:
                    for b in emission_smc["nll_per_t_per_b"][t]:
                        smc_info["log_p_y_giv_xz__per_t"][t, b, k] = -emission_smc["nll_per_t_per_b"][t][b].detach()

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
        ret["info_temporal"]["elbo_t"] = [(sum(ret["info_temporal"]["kl_t"][:t+1]) + ret["info_temporal"]["recon_nll_t"][t]) for t in range(n_timesteps)]

        # iwae nll
        ret["info_scalar"]["iwae_nll"] = -torch.mean(
            sum([
                log_mean_exp(smc_info["log_p_y_giv_xz__per_t"][t] + smc_info["log_w__per_t_per_k"][t])
            for t in range(n_timesteps)])
        ).detach().item()
        ret["info_temporal"]["iwae_nll_t"] = [-torch.mean(
            log_mean_exp(smc_info["log_p_y_giv_xz__per_t"][t]) + torch.sum(smc_info["log_mean_w__per_t"][:t+1], dim=0)
        ).detach().item() for t in range(n_timesteps)]

        # gen nll
        if eval_mse:
            SS = None
            for k in range(K):
                SS = {
                    "snp": {-1: {"cs_t": cs_t0, 'z_t': z_t0}},
                }

                for t in range(n_timesteps):
                    response = self.tconvdraw.generate(R_C[t], AA[t], SS["snp"][t - 1]["cs_t"], SS["snp"][t - 1]["z_t"])
                    SS["snp"][t] = {"cs_t": response["cs_t"], 'z_t': response["z_t"]}

                # snp gen emission
                emission_snp = self.emission(SS["snp"], XY, std=std)

                # record nll
                info_gen_nll_adder.append(emission_snp["nll"].detach())
                info_gen_nll_t_adder.append_list([emission_snp["nll_per_t"][t].detach() for t in emission_snp["nll_per_t"]])

        ret["info_scalar"]["gen_nll"] = info_gen_nll_adder.mean().detach().item()
        ret["info_temporal"]["gen_nll_t"] = [item.detach().item() for item in info_gen_nll_t_adder.mean_list()]

        return ret