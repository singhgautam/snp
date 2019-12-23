'''
miscellaneous functions: learning
'''
import os
import datetime
import copy

import numpy as np

import torch
import torchvision.utils as vutils
from torchvision import transforms

from PIL import Image, ImageDraw, ImageFont

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


''' for monitoring lr '''
def get_lrs(optimizer):
    lrs = [float(param_group['lr']) for param_group in optimizer.param_groups]
    lr_max = max(lrs)
    lr_min = min(lrs)
    return lr_min, lr_max


''' save and load '''
def save_checkpoint(state, path='', filename='checkpoint.pth.tar'):
    filename = os.path.join(path, filename)
    print("=> save checkpoint '{}'".format(filename))
    torch.save(state, filename)


def load_checkpoint(model, optimizer, path='', filename='checkpoint.pth.tar', verbose=True):
    filename = os.path.join(path, filename) 
    if os.path.isfile(filename):
        if verbose:
            print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if verbose:
            print("=> loaded checkpoint '{}'".format(filename))
    else:
        print("=> no checkpoint found at '{}'".format(filename))


''' log '''
def logging(s, path, filename='log.txt', print_=True, log_=True):
    if print_:
        print(s)
    if log_:
        with open(os.path.join(path, filename), 'a+') as f_log:
            f_log.write(s + '\n')

def get_time():
    return datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S.%f')


''' visualization '''
def get_image_from_values(input, batch_size, num_channels, num_height):
    '''
    input : b x c x h x w (where h = w = 1)
    '''
    assert num_height == 1
    input = input.detach()
    output = input.view(1, 1, batch_size, num_channels*num_height*num_height).clone().cpu()
    output = vutils.make_grid(output, normalize=True, scale_each=True)
    return output

def get_grid_image(input, batch_size, num_channels, num_height, nrow=8, pad_value=0, scale_mode=1, normalize=True):
    '''
    input : b x c x h x w (where h = w)
    '''
    input = input.detach()
    output = input.view(batch_size, num_channels, num_height, num_height).clone().cpu()

    if scale_mode == 0:
        output = vutils.make_grid(output, nrow=nrow, normalize=normalize, scale_each=False, range=(0, 1.0), pad_value=pad_value, padding=2)
    else:
        output = vutils.make_grid(output, nrow=nrow, normalize=normalize, scale_each=True, pad_value=pad_value, padding=2)
    # output = vutils.make_grid(output, nrow=nrow, normalize=True, scale_each=True, pad_value=pad_value)
    #output = vutils.make_grid(output, normalize=False, scale_each=False)
    return output

def get_grid_image_padded_rowmajor(input, grid_size, pad_value=0):
    '''
    input : Torch tensor (a1 x a2 x ... an) x c x h x w

    All images in the input will be arranged in a row major fashion
    on the grid of the given size. If there are more images, they will simply be dropped.
    If there are less images, they will be padded by blank images in the end to fill the grid up.

    '''

    input = input.detach()

    grid_w, grid_h = grid_size
    nc, nw, nh = input.size()[-3], input.size()[-2], input.size()[-1]
    input = input.view(-1, nc, nw, nh).clone().cpu()
    if len(input) < grid_w * grid_h:
        num_pad = grid_w * grid_h - len(input)
        padding = torch.zeros(num_pad, nc, nw, nh)
        input = torch.cat([input, padding], dim=0)
    else:
        input = input[:grid_h*grid_w]
    output = vutils.make_grid(input, nrow=grid_w, normalize=True, scale_each=True, pad_value=pad_value)
    return output

def get_grid_image_padded_arranged_rowwise(input, nchannels, nheight, grid_size, pad_value=0, scale_mode=1, normalize=True):
    '''
    input : List of Tensors (a1 x a2 ... x an) x c x h x w

    The grid image will contain in every row the elements of every element in the input.
    Per row, if there are more images than the grid_width, then they are simply dropped. If there
    are less, then they would simply be padded by blank images.

    '''

    grid_h, grid_w = grid_size

    input_rows = []
    for input_row in input:
        input_row = torch.zeros(grid_w, nchannels, nheight, nheight) if input_row is None else input_row
        input_row = input_row.view(-1, nchannels, nheight, nheight).clone().cpu()
        if len(input_row) < grid_w:
            num_pad = grid_w - len(input_row)
            padding = torch.zeros(num_pad, nchannels, nheight, nheight)
            input_row = torch.cat([input_row, padding], dim=0)
        else:
            input_row = input_row[:grid_w]
        input_rows += [input_row]
        if len(input_rows) >= grid_h:
            break
    input = torch.cat(input_rows, dim=0)
    if scale_mode == 0:
        output = vutils.make_grid(input, nrow=grid_w, normalize=normalize, scale_each=False, range=(0, 1.0),
                                  pad_value=pad_value, padding=2)
    else:
        output = vutils.make_grid(input, nrow=grid_w, normalize=normalize, scale_each=True, pad_value=pad_value, padding=2)
    return output



def get_plot(preds, seq_length, batch_size, num_channels, num_height, cond_length=None):
    '''
    input : s x b x c x h x w (where h = w = 1)
    '''
    assert num_height == 1
    preds = preds.detach()
    preds = preds.view(seq_length, batch_size*num_channels*num_height*num_height).clone().cpu()  # temporary

    # init
    if cond_length is not None:
        cond_length = min(cond_length, seq_length)
    else:
        cond_length = seq_length

    # convert to numpy
    preds = preds.numpy()
    #data = data.numpy()

    # plot
    scale = 0.5
    fig = plt.figure(figsize=(scale*30,scale*10))
    plt.title('Predict future values for time sequences\n(Dashlines are predicted values)', fontsize=scale*30)
    plt.xlabel('x', fontsize=scale*20)
    plt.ylabel('y', fontsize=scale*20)
    plt.xticks(fontsize=scale*20)
    plt.yticks(fontsize=scale*20)
    def draw(yi, color):
        plt.plot(np.arange(cond_length), yi[:cond_length], color, linewidth = scale*2.0)
        plt.plot(np.arange(cond_length, len(yi)), yi[cond_length:], color + ':', linewidth = scale*2.0)
    #draw(data[:, 0], 'k')
    draw(preds[:, 0], 'r')
    draw(preds[:, 1], 'g')
    draw(preds[:, 2], 'b')
    #plt.savefig('predict%d.pdf'%batch_idx)
    #plt.savefig('predict.pdf')
    #plt.close()

    # draw to canvas
    fig.canvas.draw()  # draw the canvas, cache the renderer
    image = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    # close figure
    plt.close()
    return image

def get_numpy_plot(data, title=None, xlabel='x', ylabel='y'):
    '''
    input : batch_size x seq_length
    '''
    batch_size, seq_length = data.shape

    # plot
    scale = 0.5
    fig = plt.figure(figsize=(scale*30,scale*10))
    if title is not None:
        plt.title(title, fontsize=scale*30)
    plt.xlabel(xlabel, fontsize=scale*20)
    plt.ylabel(ylabel, fontsize=scale*20)
    plt.xticks(fontsize=scale*20)
    plt.yticks(fontsize=scale*20)
    def draw(yi):
        plt.plot(np.arange(seq_length), yi, linewidth = scale*2.0)
    for i in range(batch_size):
        draw(data[i])

    # draw to canvas
    fig.canvas.draw()  # draw the canvas, cache the renderer
    image = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    # close figure
    plt.close()
    return image

def make_tensor_with_text(
        text='None',
        img_size=(64,256),
        bg_color='white',
        font_size=10,
        font_file=None,
        font_color=(0,0,0),
        text_placement=(10,10),
):
    fnt = ImageFont.truetype(font_file, font_size)
    img = Image.new('RGB', (img_size[1], img_size[0]), color=bg_color)

    d = ImageDraw.Draw(img)
    d.text(text_placement, text, font=fnt, fill=font_color)

    transform = transforms.ToTensor()
    tensorified = transform(img)

    return tensorified

class NormalizedAdder:
    def __init__(self, initial):
        self.initial = initial
        self.sum = initial
        self.count = 0.0

    def append(self, value, weight=1.0):
        if value is not None:
            self.sum += value*weight
            self.count += weight

    def mean(self):
        return self.sum/self.count if self.count > 0 else self.initial

class ScaledNormalizedAdder:
    def __init__(self, initial, scale):
        self.initial = initial
        self.sum = initial
        self.count = 0.0
        self.scale = scale

    def append(self, value, weight=1.0):
        if value is not None:
            self.sum += value*self.scale*weight
            self.count += weight

    def mean(self):
        return self.sum/self.count if self.count > 0 else self.initial

class NormalizedAdderList:
    def __init__(self, initial, size):
        self.adders = [NormalizedAdder(copy.deepcopy(initial)) for _ in range(size)]

    def __getitem__(self, item):
        return self.adders[item]

    def mean_list(self):
        return [adder.mean() for adder in self.adders]

    def append_list(self, list):
        assert len(list) == len(self.adders)
        for i,item in enumerate(list):
            self.adders[i].append(item)


