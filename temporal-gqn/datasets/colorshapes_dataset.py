import os
import math
import cv2
import copy
import numpy as np
import torch
import torch.utils.data as data
import errno
from PIL import Image
import scipy.misc as misc
from datasets.context_curriculum import StepCurriculum

###########################################################################################
# script to generate moving mnist video dataset (frame by frame) as described in
# [1] arXiv:1502.04681 - Unsupervised Learning of Video Representations Using LSTMs
#     Srivastava et al
# adopted from Tencia Lee
###########################################################################################

###########################################################################################
# Copied and modified from https://github.com/Steffen-Wolf/moving_mnist.git
###########################################################################################

color_dict = {}
# obj colors
color_dict['blue'] = np.array([0, 0, 255]) / np.float32(255)
color_dict['cyan'] = np.array([0, 255, 255]) / np.float32(255)
color_dict['red'] = np.array([255, 0, 0]) / np.float32(255)
color_dict['magenta'] = np.array([255, 0, 255]) / np.float32(255)
color_dict['green'] = np.array([0, 255, 0]) / np.float32(255)
color_dict['darkyellow'] = np.array([233, 233, 0]) / np.float32(255)
# color_set = [['blue', 'cyan']]
color_set = [['blue', 'cyan'], ['red', 'magenta'], ['green', 'darkyellow']]


# background colors
# bg_color_list = ['brown', 'lightbrown', 'yellow', 'lightyellow']
# color_dict['brown'] = np.array([102, 501, 0]) / np.float32(255)
# color_dict['lightbrown'] = np.array([204, 153, 102]) / np.float32(255)
# color_dict['yellow'] = np.array([255, 204, 0]) / np.float32(255)
# color_dict['lightyellow'] = np.array([255, 204, 153]) / np.float32(255)


# helper functions
def makedir_exist_ok(dirpath):
    """
    Python2 support for os.makedirs(.., exist_ok=True)
    """
    try:
        os.makedirs(dirpath)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise


def get_shapes(num, size):
    images = np.zeros((num, size[0], size[1], 3))

    # objs_type = np.random.randint(low=0, high=3, size=num)
    # objs_color_sampling = np.random.randint(low=0, high=2, size=num)
    # objs_color = [n%len(color_set) for n in range(num)]
    # objs_color = [color_set[oc][ocs] for oc, ocs in zip(objs_color, objs_color_sampling)]

    # 3 shapes x 3 colors
    objs = np.random.randint(low=0, high=3 * len(color_set), size=num)
    objs_color_sampling = np.random.randint(low=0, high=2, size=num)
    objs_type = np.array([o // 3 for o in objs])
    objs_color = np.array([o % len(color_set) for o in objs])
    objs_idx = np.argsort(objs_color)
    objs_type = objs_type[objs_idx]
    objs_color = objs_color[objs_idx]
    # when color set is same then lower idx is overwritten
    objs_color_list = []
    st = 0
    nd = 0
    while True:
        if nd == num:
            break
        if objs_color[st] == objs_color[nd]:
            if nd == num - 1:
                if nd - st != 1:
                    ocs_idx = np.argsort(objs_color_sampling[st:nd])
                    objs_type_tmp = objs_type[st:nd]
                    objs_type[st:nd] = objs_type_tmp[ocs_idx]
                objs_color_list.append((st, nd + 1))
            nd += 1
        else:
            if nd - st != 1:
                objs_color_sampling[st:nd].sort()
                ocs_idx = np.argsort(objs_color_sampling[st:nd])
                objs_type_tmp = objs_type[st:nd]
                objs_type[st:nd] = objs_type_tmp[ocs_idx]
                objs_color_list.append((st, nd))
                st = copy.copy(nd)
            else:
                objs_color_list.append((st, nd))
                st = copy.copy(nd)
    objs_color_str = [color_set[oc][ocs] for oc, ocs in zip(objs_color, objs_color_sampling)]

    for n, ot, oc in zip(range(num), objs_type, objs_color_str):
        color_map = color_dict[oc]
        canvas = np.ones((size[0], size[1], 3))

        # shape
        if ot == 0:  # tri
            pt1 = (size[0], 0)
            pt2 = (size[0], size[1])
            pt3 = (0, int(size[1] / 2))
            triangle_cnt = np.array([pt1, pt2, pt3])
            cv2.drawContours(canvas, [triangle_cnt], 0, color_map, -1)
            images[n] = canvas
        elif ot == 1:  # rec
            canvas = color_map
            images[n] = canvas
        elif ot == 2:  # cir
            cv2.circle(canvas, center=(int(size[0] / 2), int(size[1] / 2)), radius=int(size[0] / 2), color=color_map,
                       thickness=-1)
            images[n] = canvas
        else:
            raise ValueError('obj type ' + str(objs_type[n]) + ' is not valid')
            exit(1)
    return images, objs_color_str, objs_color_list, objs_color_sampling


def get_img_query(t, data, frame_shape, height_low, height_high, height, width_low, width_high, width, n_frame,
                  positions, digit_bias_x, digit_bias_y, encoded_t, t_query_flag=False, regular_targets=False, ngrid=3):
    if n_frame == 0:
        return None, None

    # random view
    if regular_targets:
        query_bundle = []
        for i in range(ngrid):
            for j in range(ngrid):
                xpos = height_low + (i / (ngrid - 1.))*(height_high - height_low)
                ypos = width_low + (j / (ngrid - 1.))*(width_high - width_low)
                query_bundle += [(int(xpos), int(ypos))]
    else:
        camera_pos_x = np.random.randint(low=height_low, high=height_high, size=n_frame)
        camera_pos_y = np.random.randint(low=width_low, high=width_high, size=n_frame)
        query_bundle = zip(camera_pos_x, camera_pos_y)

    queries = []
    images = []
    for cam_x, cam_y in query_bundle:
        image = data[t, cam_x: cam_x + frame_shape[0], cam_y: cam_y + frame_shape[1]]
        if t_query_flag:
            query = np.array([(1.0 * cam_x) / height, (1.0 * cam_y) / width, encoded_t])
        else:
            query = np.array([(1.0 * cam_x) / height, (1.0 * cam_y) / width])

        image = torch.from_numpy(image).float()
        image = image.permute(2, 0, 1)
        image = image.view(1, image.shape[0], image.shape[1], image.shape[2])

        query = torch.from_numpy(query).float()
        query = query.view(1, -1)

        images.append(image)
        queries.append(query)

    images = torch.cat(images, dim=0)
    queries = torch.cat(queries, dim=0)

    return images, queries


class MovingColorShapesDataset(data.Dataset):

    def __init__(self, root='data', img_size=(96, 96), seq_len=20, nums_per_image=2, frame_shape=(64, 64),
                 fake_dataset_size=1000, num_views=20,
                 allow_empty_context=False, target_sample_method='remaining', min_cond_size=4, max_cond_size=20,
                 max_target_size=20, curriculum=StepCurriculum(), max_vel=15, min_vel=15, color_change_flag=True,
                 ratio_size_flag=False, t_query_flag=False, regular_targets_info=False, ngrid=3):
        self.root = root
        self.epoch_length = fake_dataset_size
        self.frame_shape = frame_shape
        if ratio_size_flag:
            self.obj_shape = (int(frame_shape[0] * 0.6), int(frame_shape[1] * 0.6))
            self.img_shape = (int(self.obj_shape[0] * math.sqrt(nums_per_image * 5.9)),
                              int(self.obj_shape[1] * math.sqrt(nums_per_image * 5.9)))
        else:
            self.img_shape = img_size
            self.obj_shape = (28, 28)
        self.x_lim = self.img_shape[0] - self.obj_shape[0]
        self.y_lim = self.img_shape[1] - self.obj_shape[1]
        self.lims = (self.x_lim, self.y_lim)
        self.seq_len = seq_len
        self.num_views = num_views
        self.allow_empty_context = allow_empty_context
        self.target_sample_method = target_sample_method
        self.min_cond_size = min_cond_size
        self.max_cond_size = max_cond_size
        self.max_target_size = max_target_size
        self.curriculum = curriculum
        self.max_vel = max_vel
        self.min_vel = min_vel
        self.nums_per_image = nums_per_image
        self.rand_seed = np.random.randint(low=0, high=1000000, size=[self.epoch_length])
        self.color_change_prob = 1 / self.seq_len
        self.color_change_flag = color_change_flag
        self.t_query_flag = t_query_flag
        self.regular_targets_info = regular_targets_info
        self.ngrid = ngrid

    def __len__(self):
        return self.epoch_length

    def __getitem__(self, index):
        # randomly generate direc/speed/position, calculate velocity vector
        np.random.seed(self.rand_seed[index])  # seed setting
        direcs = np.pi * (np.random.rand(self.nums_per_image) * 2 - 1)
        speeds = np.random.randint(low=self.min_vel, high=self.max_vel + 1, size=self.nums_per_image)
        veloc = [(v * math.cos(d), v * math.sin(d)) for d, v in zip(direcs, speeds)]
        images, colors, objs_color_list, objs_color_sampling = get_shapes(self.nums_per_image, self.obj_shape)
        # bg_color = np.array([0, 0, 0]) / np.float32(255)
        bg_color = np.array([255, 255, 255]) / np.float32(255)
        positions = [(np.random.rand() * self.x_lim, np.random.rand() * self.y_lim) for _ in range(self.nums_per_image)]

        if self.regular_targets_info:
            n_grid = self.ngrid
            regular_positions = []
            for i in range(n_grid):
                for j in range(n_grid):
                    xpos = i / (n_grid - 1.)
                    ypos = j / (n_grid - 1.)
                    regular_positions += [(xpos, ypos)]

        data = np.ones((self.seq_len, self.img_shape[0], self.img_shape[1], 3))

        position_list = []
        color_change_history = [False for n in range(self.nums_per_image)]
        for frame_idx in range(self.seq_len):
            if self.color_change_flag:
                color_change_check = np.random.uniform(low=0.0, high=1.0, size=self.nums_per_image)
                for n in range(self.nums_per_image):
                    if color_change_history[n]:
                        continue
                    if color_change_check[n] < self.color_change_prob:
                        color = colors[n]
                        for cs in color_set:
                            if color in cs:
                                objs_color_sampling[n] = (cs.index(color) + 1) % 2
                                color = cs[objs_color_sampling[n]]
                                break
                        colors[n] = color
                        color_map = color_dict[color]
                        for i in range(self.obj_shape[0]):
                            for j in range(self.obj_shape[1]):
                                if np.sum(images[n][i, j]) != 3.0:
                                    images[n][i, j] = color_map
                        color_change_history[n] = True

            # re-order
            for st, nd in objs_color_list:
                if nd != st + 1:
                    ocs_idx = np.argsort(objs_color_sampling[st:nd])
                    images_tmp = images[st:nd]
                    images[st:nd] = images_tmp[ocs_idx]
                    ocs_tmp = objs_color_sampling[st:nd]
                    objs_color_sampling[st:nd] = ocs_tmp[ocs_idx]
                    pos_tmp = np.array(positions[st:nd])
                    positions[st:nd] = pos_tmp[ocs_idx]
                    color_tmp = np.array(colors[st:nd])
                    colors[st:nd] = color_tmp[ocs_idx]
                    cch_tmp = np.array(color_change_history[st:nd])
                    color_change_history[st:nd] = cch_tmp[ocs_idx]

            position_list.append(positions)
            for i, obj in enumerate(images):
                x, y = int(positions[i][0]), int(positions[i][1])
                for ii in range(self.obj_shape[0]):
                    for jj in range(self.obj_shape[1]):
                        if np.sum(obj[ii, jj]) != 3.0:
                            data[frame_idx, x + ii, y + jj] = obj[ii, jj]

            # update positions based on velocity
            next_pos = [list(map(sum, zip(p, v))) for p, v in zip(positions, veloc)]
            # bounce off wall if a we hit one
            for i, pos in enumerate(next_pos):
                # bound the side of canvas
                for j, coord in enumerate(pos):
                    if coord < 0 or coord > self.lims[j]:
                        veloc[i] = tuple(list(veloc[i][:j]) + [-1 * veloc[i][j]] + list(veloc[i][j + 1:]))
            positions = [list(map(sum, zip(p, v))) for p, v in zip(positions, veloc)]

        data = np.clip(data, 0, 1)
        data = data.swapaxes(1, 2)

        images_context = []
        queries_context = []
        images_target = []
        queries_target = []
        info = {}
        info['scene_maps'] = []
        info['scene_maps_original'] = []
        info['regular_images'] = []
        num_context_curriculum = self.curriculum.get(self.seq_len)

        for t in range(self.seq_len):
            info['scene_maps'] += [
                torch.from_numpy(
                    np.array(Image.fromarray(np.uint8(data[t]*255)).resize(self.frame_shape))
                ).permute(2, 0, 1).float()/255.
            ]

            info['scene_maps_original'] += [
                torch.from_numpy(data[t]).permute(2, 0, 1).float()
            ]

            # get the context and target sizes
            context_size = num_context_curriculum[t]
            if self.target_sample_method == "remaining":
                target_size = self.num_views - context_size
            elif self.target_sample_method == "full":
                target_size = self.num_views


            # add context frames
            height_high = int(self.img_shape[0]) - self.frame_shape[0]
            height_low = 0
            width_high = int(self.img_shape[1]) - self.frame_shape[1]
            width_low = 0
            context_img, context_query = get_img_query(t, data, self.frame_shape, height_low, height_high,
                                                       self.img_shape[0] - self.frame_shape[0], width_low,
                                                       width_high, self.img_shape[1] - self.frame_shape[1],
                                                       context_size, position_list[t], int(self.obj_shape[0] / 2),
                                                       int(self.obj_shape[1] / 2), 0.25 + 0.5 * t / self.seq_len,
                                                       t_query_flag=self.t_query_flag)
            images_context.append(context_img)
            queries_context.append(context_query)

            # get target frames
            target_img, target_query = get_img_query(t, data, self.frame_shape, 0,
                                                     self.img_shape[0] - self.frame_shape[0],
                                                     self.img_shape[0] - self.frame_shape[0], 0,
                                                     self.img_shape[1] - self.frame_shape[1],
                                                     self.img_shape[1] - self.frame_shape[1], target_size,
                                                     position_list[t], int(self.obj_shape[0] / 2),
                                                     int(self.obj_shape[1] / 2), 0.25 + 0.5 * t / self.seq_len,
                                                     t_query_flag=self.t_query_flag, regular_targets=self.regular_targets_info, ngrid=self.ngrid)
            images_target.append(target_img)
            queries_target.append(target_query)

        return (images_context, queries_context), (images_target, queries_target), (None), (info)


def collate_fn(batches):
    '''
    Input:
      batch: list of tuples, each of which is
             (context, target)
             where context = (images, cameras)
                   target  = (images, cameras)
    Output:
      contexts: a list, whose element is context (a list over batches) at a certain time-step
                where context = [(images, cameras)]
      targets:  a list, whose element is target (a list over batches) at a certain time-step
                where context = [(images, cameras)]
    '''

    # number of timesteps
    num_timesteps = len(batches[0][0][0])

    # init
    contexts = [[(context[0][t], context[1][t]) for context, _, _, _ in batches] for t in range(num_timesteps)]
    targets = [[(target[0][t], target[1][t]) for _, target, _, _ in batches] for t in range(num_timesteps)]
    info = [_info for _, _, _, _info in batches]

    return contexts, targets, None, info


def get_color_shapes_scene_dataset(
        train_batch_size,
        eval_batch_size,
        kwargs,
        allow_empty_context=False,
        target_sample_method='remaining',
        min_cond_size=5,
        max_cond_size=20,
        max_target_size=20,
        num_timesteps=10,
        num_views=10,
        img_size=(96,96),
        curriculum=StepCurriculum(init_num_views=5),
        max_vel=13,
        min_vel=13,
        nums_per_image=2,
        frame_size=64,
        color_change_flag=True,
        ratio_size_flag=True,
        t_query_flag=False,
        regular_targets_info=False,
        ngrid=3,

):
    # init dataset (train / val)
    train_dataset = MovingColorShapesDataset(
        root='data',
        frame_shape=(frame_size, frame_size),
        allow_empty_context=allow_empty_context,
        target_sample_method=target_sample_method,
        fake_dataset_size=800000,
        min_cond_size=min_cond_size,
        max_cond_size=max_cond_size,
        max_target_size=max_target_size,
        seq_len=num_timesteps,
        num_views=num_views,
        img_size=img_size,
        curriculum=curriculum,
        max_vel=max_vel,
        min_vel=min_vel,
        nums_per_image=nums_per_image,
        color_change_flag=color_change_flag,
        ratio_size_flag=ratio_size_flag,
        t_query_flag=t_query_flag,
        regular_targets_info=regular_targets_info,
        ngrid=ngrid,
    )
    val_dataset = MovingColorShapesDataset(
        root='data',
        frame_shape=(frame_size, frame_size),
        allow_empty_context=allow_empty_context,
        target_sample_method=target_sample_method,
        fake_dataset_size=20000,
        min_cond_size=min_cond_size,
        max_cond_size=max_cond_size,
        max_target_size=max_target_size,
        seq_len=num_timesteps,
        num_views=num_views,
        img_size=img_size,
        curriculum=curriculum,
        max_vel=max_vel,
        min_vel=min_vel,
        nums_per_image=nums_per_image,
        color_change_flag=color_change_flag,
        ratio_size_flag=ratio_size_flag,
        t_query_flag=t_query_flag,
        regular_targets_info=regular_targets_info,
        ngrid=ngrid,
    )
    test_dataset = MovingColorShapesDataset(
        root='data',
        frame_shape=(frame_size, frame_size),
        allow_empty_context=allow_empty_context,
        target_sample_method=target_sample_method,
        fake_dataset_size=20000,
        min_cond_size=min_cond_size,
        max_cond_size=max_cond_size,
        max_target_size=max_target_size,
        seq_len=num_timesteps,
        num_views=num_views,
        img_size=img_size,
        curriculum=curriculum,
        max_vel=max_vel,
        min_vel=min_vel,
        nums_per_image=nums_per_image,
        color_change_flag=color_change_flag,
        ratio_size_flag=ratio_size_flag,
        t_query_flag=t_query_flag,
        regular_targets_info=regular_targets_info,
        ngrid=ngrid,
    )

    # init dataloader
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=train_batch_size, shuffle=True, collate_fn=collate_fn,
                                               **kwargs)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=train_batch_size, shuffle=False, collate_fn=collate_fn,
                                             **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=eval_batch_size, shuffle=False, collate_fn=collate_fn,
                                              **kwargs)

    # init info
    info = {}
    info['nviews'] = train_dataset.num_views
    info['ntimesteps'] = train_dataset.seq_len
    info['max_cond_size'] = train_dataset.max_cond_size
    info['max_target_size'] = train_dataset.max_target_size
    info['allow_empty_context'] = train_dataset.allow_empty_context
    info['target_sample_method'] = train_dataset.target_sample_method
    info['max_vel'] = max_vel
    info['min_vel'] = min_vel

    return train_loader, val_loader, test_loader, info