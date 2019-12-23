# import
import os
import argparse
import time

from tensorboardX import SummaryWriter

from configure_args import configure_argparser

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch import nn

import models as net

from utils import get_time, logging, get_lrs, tensorboardlogger, Illustrator, lists_to_tikz
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
    opt.experiment = '{}_{}_lr-{}_{}'.format(
                      opt.dataset,
                      opt.model,
                      opt.lr,
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
    loaded_model = torch.load(opt.load_model)
    print('Loaded model from {}'.format(opt.load_model))
    if opt.state_dict:
        model_state_dict = model.state_dict()
        loaded_model_state_dict = {k: copy_state(v, model_state_dict[k]) for k, v in loaded_model.state_dict().items()
                                   if k in model_state_dict}
        model_state_dict.update(loaded_model_state_dict)
        model.load_state_dict(model_state_dict)
    else:
        model = loaded_model

# init optimizer
if opt.optimizer == 'sgd':
    optimizer = optim.SGD(model.parameters(), lr=opt.lr)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=1./4.0, patience=0, verbose=True)
elif opt.optimizer == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    scheduler = None
else:
    raise NotImplementedError('unknown optimizer: {}'.format(opt.optimizer))

# log opt
opt.pid = os.getpid()
tensorboardlogger.log_class_attributes(opt, "process/options")

# define train
def train(epoch):
    model.train()
    epoch_info_adder = NormalizedDictAdder(initial=0)
    running_info_adder = NormalizedDictAdder(initial=0)
    start_time = time.time()
    for batch_idx, data_batch in enumerate(train_loader):
        # total iterations
        total_iterations = epoch * len(train_loader) + batch_idx

        # std and beta
        std = opt.std
        beta = opt.beta

        # data batch
        C, XY, A, data_info = data_batch[:4]

        # send to primary device
        C = recursive_to_device(C, opt.cuda_device[0])
        XY = recursive_to_device(XY, opt.cuda_device[0])
        if A:
            A = recursive_to_device(A, opt.cuda_device[0])

        # init numbers
        ntimesteps = len(C)
        num_episodes = len(C[0])

        # init grad
        model.zero_grad()

        if opt.data_parallel and num_episodes >= len(opt.cuda_device):
            # model replica(s)
            replicas = nn.parallel.replicate(model, opt.cuda_device)

            # scatter data set
            scat_C = scatter(C, opt.cuda_device)
            scat_XY = scatter(XY, opt.cuda_device)
            if A:
                scat_A = scatter(A, opt.cuda_device)

            # zip scattered data
            train_input = [ (
                    scat_C[dev_id],
                    scat_XY[dev_id],
                    scat_A[dev_id] if A else None,
                    {"beta":beta, "std":std}
                ) for dev_id in range(len(opt.cuda_device))]

            # run model
            scat_model_output = nn.parallel.parallel_apply(replicas, train_input)
            model_output = collect_scattered_outputs(scat_model_output, opt.cuda_device[0])

        else:
            model_output = model(C, XY, A, {"beta":beta, "std":std})

        epoch_info_adder.append(model_output)
        running_info_adder.append(model_output)

        loss = model_output["__loss__"]
        loss.backward()

        if opt.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), opt.clip)

        optimizer.step()

        # running print
        if batch_idx % opt.log_interval == 0:
            model.train()

            # set log info
            elapsed = time.time() - start_time
            start_time = time.time()

            lr_min, lr_max = get_lrs(optimizer)

            # print
            logging('| epoch {:3d} | {:5d}/{:5d} '
                    '| lr_min {:02.4f} | lr_max {:02.4f} | ms/step {:5.2f} '
                    '| loss {:5.8f} | '
                    .format(
                        epoch,
                        batch_idx+1, len(train_loader),
                        lr_min, lr_max, elapsed * 1000 / opt.log_interval,
                        model_output["info_scalar"]["loss"],
                    ), path=opt.path)

            # tensorboard
            for scalar in running_info_adder.mean()["info_scalar"]:
                writer.add_scalar('train/{}/step'.format(scalar), running_info_adder.mean()["info_scalar"][scalar], total_iterations)

            # tensorboard temporal
            for temporal in running_info_adder.mean()["info_temporal"]:
                data_list = running_info_adder.mean()["info_temporal"][temporal]
                writer.add_text("train/{}/step".format(temporal), lists_to_tikz(
                    list(range(len(data_list))),
                    data_list
                ), total_iterations)

            # reset
            running_info_adder = NormalizedDictAdder(initial=0)

        # visualize and save model
        if batch_idx == 0 or (batch_idx % opt.vis_interval) == 0 or (batch_idx+1 == len(train_loader)):
            with torch.no_grad():
                X = [[XY[t][b][QUERIES] for b in range(len(XY[t]))] for t in range(len(XY))]
                gen_output = model.generate(C, X, A)
            model.train()

            for b in range(min(num_episodes,4)):
                map_tile = [[data_info[b]["scene_maps"][t].cpu()] for t in range(len(XY))]
                A_tile = [[data_info[b]["actions"][t].cpu()] for t in range(len(XY))] if "actions" in data_info[b] else [[]]
                C_tile = [[image.cpu() for image in C[t][b][IMAGES]] if C[t][b][IMAGES] is not None else [] for t in range(len(C))]
                Y_tile = [[image.cpu() for image in XY[t][b][IMAGES]] if XY[t][b][IMAGES] is not None else [] for t in range(len(XY))]

                vis_data = {
                    "map": map_tile,
                    "A": A_tile,
                    "C": C_tile,
                    "D": Y_tile,
                }

                # reconstruction data
                for category in model_output["__mu_Y__"]:
                    mu_Y_tile = [[image.cpu() for image in model_output["__mu_Y__"][category][t][b]] if model_output["__mu_Y__"][category][t][b] is not None
                                    else [] for t in model_output["__mu_Y__"][category]]
                    template_label = "__mu_Y__"+category+"_recon"
                    vis_data[template_label] = mu_Y_tile

                # generation data
                for category in gen_output["__mu_Y__"]:
                    mu_Y_tile = [[image.cpu() for image in gen_output["__mu_Y__"][category][t][b]] if gen_output["__mu_Y__"][category][t][b] is not None
                                    else [] for t in gen_output["__mu_Y__"][category]]
                    template_label = "__mu_Y__"+category+"_gen"
                    vis_data[template_label] = mu_Y_tile

                visual = illustrator.illustrate(vis_data)
                writer.add_image('train/CXY-recon-gen/{}'.format(b), visual, total_iterations)

            # save model
            with open(os.path.join(opt.path, 'model.pt'), 'wb') as f:
                torch.save(model, f)

        # checkpoint model
        if (total_iterations % opt.ckpt_interval) == 0 or total_iterations == 0:
            with open(os.path.join(opt.path, 'model_{}_{}.pt'.format(total_iterations, get_time())), 'wb') as f:
                torch.save(model, f)

        # end of epoch
        if batch_idx+1 == len(train_loader):
            # print
            logging('| EPOCH END {:3d} | {:5d}/{:5d} batches | loss {:5.8f} |'
                    .format(
                        epoch,
                        batch_idx+1, len(train_loader),
                        epoch_info_adder.mean()["info_scalar"]["loss"]),
                    path=opt.path)

            # write to tensorboard
            for scalar in epoch_info_adder.mean()["info_scalar"]:
                writer.add_scalar('train/{}/epoch'.format(scalar), epoch_info_adder.mean()["info_scalar"][scalar], total_iterations)

try:
    for epoch in range(opt.epochs):
        train(epoch)
except KeyboardInterrupt:
    logging('-' * 89, path=opt.path)
    logging('Exiting from training early', path=opt.path)
