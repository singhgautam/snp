# import
import os
import argparse
import time

from tensorboardX import SummaryWriter

from configure_args import configure_argparser

import torch

import models as net

from utils import get_time, logging, tensorboardlogger, Illustrator, lists_to_tikz
from utils.train_tools import *

from datasets.context_curriculum import ManualCurriculumRandom
from datasets.colorshapes_dataset import get_color_shapes_scene_dataset

# MACROS
IMAGES = 0
QUERIES = 1

# parse arguments
parser = argparse.ArgumentParser()
configure_argparser(parser)

# parse arguments
opt = parser.parse_args()

# preprocess arguments
opt.cuda = not opt.no_cuda and torch.cuda.is_available()
device = torch.device("cuda:{}".format(opt.cuda_device[0]) if opt.cuda else "cpu")

print("Using device: ", device)

# check arguments
opt.inputsize = opt.nheight * opt.nheight * opt.nchannels

# generate cache folder
if opt.cache is None:
    opt.cache = 'experiments'
if opt.experiment is None:
    opt.experiment = 'eval_{}_{}_{}'.format(
                      opt.dataset,
                      opt.model,
                      get_time())
opt.path = os.path.join(opt.cache, opt.experiment)
os.system('mkdir -p {}'.format(opt.path))

# print args
logging(str(opt),  path=opt.path)

# init tensorboard
writer = SummaryWriter(opt.path)
tensorboardlogger.writer = writer

curriculum = ManualCurriculumRandom(
    upper_bound=opt.manual_curriculum_bound,
    allow_empty_context=opt.allow_empty_context,
)

# data parallel
opt.data_parallel = False
if len(opt.cuda_device) > 1:
    opt.data_parallel = True

# illustrator
illustrator = Illustrator(opt.vis_template)

# init dataset
kwargs = {'num_workers': 0, 'pin_memory': True} if opt.cuda else {}
if opt.dataset in ['colorshapes']:
    train_loader, val_loader, test_loader, dataset_info = get_color_shapes_scene_dataset(
        train_batch_size=opt.train_batch_size,
        eval_batch_size=opt.eval_batch_size,
        kwargs=kwargs,
        allow_empty_context=False,
        target_sample_method="remaining",
        max_cond_size=20,
        max_target_size=20,
        curriculum=curriculum,
        num_views=opt.num_views,
        nums_per_image=opt.moving_color_shapes_num_objs,
        img_size=(opt.moving_color_shapes_canvas_size, opt.moving_color_shapes_canvas_size),
        num_timesteps=opt.num_timesteps,
    )
else:
    raise NotImplementedError("Not implemented dataset {}".format(opt.dataset))

# init model
if opt.model == 'tgqn-pd':
    model = net.TGQN_PD(
        im_height=opt.nheight,
        im_channels=opt.nchannels,
        nc_enc=opt.nc_enc,
        nc_lstm=opt.nc_lstm,
        nc_context=opt.nc_context,
        nc_query=opt.query_size,
        nc_z=opt.nz,
        n_draw_steps=opt.num_draw_steps,
        n_timesteps=opt.num_timesteps,
        nc_ssm=opt.sssm_num_state,
        loss_type=opt.recon_loss,
        context_type=opt.context_type,
        n_actions=opt.num_actions,
        shared_core=opt.shared_core,
        concat_latents=opt.concatenate_latents,
        use_ssm_context=opt.use_ssm_context,
        q_bernoulli=opt.q_bernoulli_pick,
        pd=True,
    ).to(device)
elif opt.model == 'tgqn':
    model = net.TGQN_PD(
        im_height=opt.nheight,
        im_channels=opt.nchannels,
        nc_enc=opt.nc_enc,
        nc_lstm=opt.nc_lstm,
        nc_context=opt.nc_context,
        nc_query=opt.query_size,
        nc_z=opt.nz,
        n_draw_steps=opt.num_draw_steps,
        n_timesteps=opt.num_timesteps,
        nc_ssm=opt.sssm_num_state,
        loss_type=opt.recon_loss,
        context_type=opt.context_type,
        n_actions=opt.num_actions,
        shared_core=opt.shared_core,
        concat_latents=opt.concatenate_latents,
        use_ssm_context=opt.use_ssm_context,
        q_bernoulli=opt.q_bernoulli_pick,
        pd=False,
    ).to(device)
elif opt.model == 'gqn':
    model = net.GQN(
        im_height=opt.nheight,
        im_channels=opt.nchannels,
        nc_enc=opt.nc_enc,
        nc_lstm=opt.nc_lstm,
        nc_context=opt.nc_context,
        nc_query=opt.query_size,
        nc_z=opt.nz,
        n_draw_steps=opt.num_draw_steps,
        n_timesteps=opt.num_timesteps,
        loss_type=opt.recon_loss,
        n_actions=opt.num_actions,
        action_emb_size=opt.nc_action_emb,
        shared_core=opt.shared_core,
        concat_latents=opt.concatenate_latents,
    ).to(device)
else:
    raise NotImplementedError('unknown model: {}'.format(opt.model))


# load model
if opt.load_model:
    loaded_model = torch.load(opt.load_model, map_location=device)
    print('Loaded model from {}'.format(opt.load_model))
    if opt.state_dict:
        model_state_dict = model.state_dict()
        loaded_model_state_dict = {k: copy_state(v, model_state_dict[k]) for k, v in loaded_model.state_dict().items()
                                   if k in model_state_dict}
        model_state_dict.update(loaded_model_state_dict)
        model.load_state_dict(model_state_dict)
    else:
        model = loaded_model

# log opt
opt.pid = os.getpid()
tensorboardlogger.log_class_attributes(opt, "process/options")

# define train
def evaluate():
    model.eval()
    running_info_adder = NormalizedDictAdder(initial=0)
    start_time = time.time()
    for batch_idx, data_batch in enumerate(test_loader):
        # total iterations
        total_iterations = batch_idx

        # std and beta
        std = opt.std
        beta = opt.beta
        K = opt.iwae_k

        # data batch
        C, XY, A, data_info = data_batch[:4]

        # send to primary device
        C = recursive_to_device(C, opt.cuda_device[0])
        XY = recursive_to_device(XY, opt.cuda_device[0])
        if A:
            A = recursive_to_device(A, opt.cuda_device[0])

        model_output = model.evaluate(C, XY, A, {
            "beta":beta,
            "std":std,
            "K":K,
            "eval_nll": (not opt.disable_eval_nll),
            "eval_mse": (not opt.disable_eval_mse)
        })
        running_info_adder.append(model_output)

        # running print
        if batch_idx % opt.log_interval == 0 or (batch_idx+1) == len(test_loader):
            # set log info
            elapsed = time.time() - start_time
            start_time = time.time()

            # print
            logging('| {:5d}/{:5d} '
                    '| ms/step {:5.2f} '
                    '| ELBO {:5.8f} | '
                    .format(
                        batch_idx+1, len(train_loader), elapsed * 1000 / opt.log_interval,
                        running_info_adder.mean()["info_scalar"]["elbo"],
                    ), path=opt.path)

            # tensorboard scalar
            for scalar in running_info_adder.mean()["info_scalar"]:
                writer.add_scalar('val/{}'.format(scalar), running_info_adder.mean()["info_scalar"][scalar], total_iterations)

            # tensorboard temporal
            for temporal in running_info_adder.mean()["info_temporal"]:
                data_list = running_info_adder.mean()["info_temporal"][temporal]
                writer.add_text("val/{}".format(temporal), lists_to_tikz(
                    list(range(len(data_list))),
                    data_list
                ), total_iterations)

try:
    evaluate()
except KeyboardInterrupt:
    logging('-' * 89, path=opt.path)
    logging('Exiting from evaluation early', path=opt.path)
