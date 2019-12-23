import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class ContextNetwork(nn.Module):
    def __init__(self,
                 repnet,
                 train_init_representation=False,
                 use_mean=False):
        super().__init__()
        # init
        self.train_init_representation = train_init_representation
        self.use_mean = use_mean

        # define networks
        self.repnet = repnet

        # information from representation network
        num_hidden, height, width = self.repnet.get_output_size()
        self.num_hidden = num_hidden
        self.height = height
        self.width = width

        # initial representation
        if self.train_init_representation:
            self.init_representation = Parameter(torch.zeros(1, self.num_hidden, self.height, self.width))

    def get_init_representation(self, batch_size):
        if self.train_init_representation:
            return self.init_representation.expand(batch_size, self.num_hidden, self.height, self.width)
        else:
            weight = next(self.parameters())
            return weight.new_zeros(batch_size, self.num_hidden, self.height, self.width)

    def forward(self, contexts):
        '''
        Input:
            contexts: a list, whose element is context
                      where context = (image, camera)
        Output:
            representations = num_episodes x num_channels x num_height x num_width
        '''
        num_episodes = len(contexts)

        # init representation
        representations = []

        # run representation network
        for context in contexts:
            # unpack context
            image, camera = context

            # add init_representation
            init_representation = self.get_init_representation(1)

            # forward representation network (each context)
            if image is None:
                # pass if context is empty
                representation = init_representation
            else:
                representation = self.repnet(image, camera)
                representation = torch.cat([representation, init_representation], dim=0)

            # sum over batch
            if self.use_mean:
                representation = torch.mean(representation, dim=0, keepdim=True)
            else:
                representation = torch.sum(representation, dim=0, keepdim=True)

            # append to representations
            representations += [representation]

        # concat representations
        representations = torch.cat(representations, dim=0) if len(representations) > 0 else 0

        return representations



class BiConvLSTM(nn.Module):
    '''
    Generate a convolutional BiLSTM Encoder.
    Input: a sequence
    Output: a sequence of hidden states and output states encoding the input sequence.
    copied and modified from https://github.com/Atcold/pytorch-CortexNet/blob/master/model/ConvLSTMCell.py
    '''
    def __init__(self, input_size, hidden_size, hidden_height=None, hidden_width=None, kernel_size=5, stride=1, padding=2,
                 train_init_state=False):
        super().__init__()
        hidden_size_fwd = hidden_size//2
        hidden_size_bwd = hidden_size - hidden_size_fwd
        self.fwd_rnn = ConvLSTMCell(input_size, hidden_size_fwd, kernel_size, stride, padding,
                                    train_init_state, hidden_height, hidden_width)
        self.bwd_rnn = ConvLSTMCell(input_size, hidden_size_bwd, kernel_size, stride, padding,
                                    train_init_state, hidden_height, hidden_width)
        self.height, self.width = hidden_height, hidden_width

    def forward(self, sequence):
        """
        Executes encoding the sequence
        :param sequence: a list of b x input_size x some_w x some_h
        :return: output sequence: a list of x b x 2*hidden_size x hidden_width x hidden_height
        """

        seq_len = len(sequence)
        batch_size = sequence[0].size(0)
        state_fwd = self.fwd_rnn.init_state(batch_size, [self.height, self.width])
        state_bwd = self.bwd_rnn.init_state(batch_size, [self.height, self.width])

        outputs_fwd = []
        outputs_bwd = []
        for t in range(seq_len):
            t_fwd, t_bwd = t, seq_len - t - 1
            output_fwd, state_fwd = self.fwd_rnn(sequence[t_fwd], state_fwd)
            output_bwd, state_bwd = self.bwd_rnn(sequence[t_bwd], state_bwd)

            outputs_fwd += [output_fwd]
            outputs_bwd += [output_bwd]

        outputs_bwd.reverse()
        output = [torch.cat([outputs_fwd[t], outputs_bwd[t]], dim=1) for t in range(seq_len)]

        return output


class ConvLSTMBackwardEncoder(nn.Module):
    '''
    Generate a convolutional encoder that goes in backward direction.
    Input: a sequence
    Output: a sequence of hidden states and output states encoding the input sequence.
    copied and modified from https://github.com/Atcold/pytorch-CortexNet/blob/master/model/ConvLSTMCell.py
    '''

    def __init__(self, input_size, hidden_size, hidden_height=None, hidden_width=None, kernel_size=5, stride=1,
                 padding=2,
                 train_init_state=False):
        super().__init__()
        hidden_size_bwd = hidden_size

        self.bwd_rnn = ConvLSTMCell(input_size, hidden_size_bwd, kernel_size, stride, padding,
                                    train_init_state, hidden_height, hidden_width)
        self.height, self.width = hidden_height, hidden_width

    def forward(self, sequence):
        """
        Executes encoding the sequence
        :param sequence: a list of b x input_size x some_w x some_h
        :return: output sequence: a list of x b x 2*hidden_size x hidden_width x hidden_height
        """
        seq_len = len(sequence)
        batch_size = sequence[0].size(0)
        state_bwd = self.bwd_rnn.init_state(batch_size, [self.height, self.width])

        outputs_bwd = []
        for t in range(seq_len):
            t_bwd = seq_len - t - 1
            output_bwd, state_bwd = self.bwd_rnn(sequence[t_bwd], state_bwd)
            outputs_bwd += [output_bwd]

        outputs_bwd.reverse()
        return outputs_bwd



class ConvLSTMForwardEncoder(nn.Module):
    '''
    Generate a convolutional encoder that goes in backward direction.
    Input: a sequence
    Output: a sequence of hidden states and output states encoding the input sequence.
    copied and modified from https://github.com/Atcold/pytorch-CortexNet/blob/master/model/ConvLSTMCell.py
    '''

    def __init__(self, input_size, hidden_size, hidden_height=None, hidden_width=None, kernel_size=5, stride=1,
                 padding=2,
                 train_init_state=False):
        super().__init__()
        hidden_size_fwd = hidden_size

        self.fwd_rnn = ConvLSTMCell(input_size, hidden_size_fwd, kernel_size, stride, padding,
                                    train_init_state, hidden_height, hidden_width)
        self.height, self.width = hidden_height, hidden_width

    def forward(self, sequence):
        """
        Executes encoding the sequence
        :param sequence: a list of b x input_size x some_w x some_h
        :return: output sequence: a list of x b x 2*hidden_size x hidden_width x hidden_height
        """
        seq_len = len(sequence)
        batch_size = sequence[0].size(0)
        state_fwd = self.fwd_rnn.init_state(batch_size, [self.height, self.width])

        outputs_fwd = []
        for t in range(seq_len):
            output_fwd, state_fwd = self.fwd_rnn(sequence[t], state_fwd)
            outputs_fwd += [output_fwd]

        return outputs_fwd



class LSTMForwardEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, train_init_state=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.LSTMCell(input_size, hidden_size)
        self.train_init_state=train_init_state

        if self.train_init_state:
            self.init_hidden = Parameter(torch.zeros(1, self.hidden_size))
            self.init_cell = Parameter(torch.zeros(1, self.hidden_size))

    def init_state(self, batch_size):
        if self.train_init_state:
            return (self.init_hidden.repeat(batch_size,1),
                    self.init_cell.repeat(batch_size,1))
        else:
            weight = next(self.parameters())
            return (weight.new_zeros(batch_size, self.hidden_size),
                    weight.new_zeros(batch_size, self.hidden_size))

    def forward(self, sequence):
        seq_len = len(sequence)
        batch_size = sequence[0].size(0)
        state_fwd = self.init_state(batch_size)

        outputs_fwd = []
        for t in range(seq_len):
            state_fwd = self.rnn(sequence[t], state_fwd)
            outputs_fwd += [state_fwd[0]]

        return outputs_fwd



class ConvLSTMCell(nn.Module):
    '''
    Generate a convolutional LSTM cell
    copied and modified from https://github.com/Atcold/pytorch-CortexNet/blob/master/model/ConvLSTMCell.py
    '''
    def __init__(self, input_size, hidden_size, kernel_size=5, stride=1, padding=2, train_init_state=False, height=None, width=None):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.train_init_state = train_init_state
        self.height = height
        self.width = width

        # lstm gates
        self.gates = nn.Conv2d(input_size + hidden_size, 4 * hidden_size, kernel_size=kernel_size, stride=stride, padding=padding)

        # initial states
        if self.train_init_state:
            assert self.height and self.width
            self.init_hidden = Parameter(torch.zeros(1, self.hidden_size, self.height, self.width))
            self.init_cell   = Parameter(torch.zeros(1, self.hidden_size, self.height, self.width))

    def init_state(self, batch_size, spatial_size):
        state_size = [batch_size, self.hidden_size] + list(spatial_size)
        if self.train_init_state:
            return (self.init_hidden.expand(state_size),
                    self.init_cell.expand(state_size))
        else:
            weight = next(self.parameters())
            return (weight.new_zeros(state_size),
                    weight.new_zeros(state_size))

    def forward(self, input, prev_state):
        # get batch and spatial sizes
        batch_size = input.data.size(0)
        spatial_size = input.data.size()[2:]

        # generate empty prev_state, if None is provided
        if prev_state is None:
            prev_state = self.init_state(batch_size, spatial_size)

        prev_hidden, prev_cell = prev_state

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat((input, prev_hidden), 1)
        outputs = self.gates(stacked_inputs)

        # chunk across channel dimension
        in_gate, remember_gate, out_gate, cell_gate = outputs.chunk(4, 1)

        # apply sigmoid non linearity
        in_gate = F.sigmoid(in_gate)
        remember_gate = F.sigmoid(remember_gate)
        out_gate = F.sigmoid(out_gate)

        # apply tanh non linearity
        cell_gate = F.tanh(cell_gate)

        # compute current cell and hidden state
        cell = (remember_gate * prev_cell) + (in_gate * cell_gate)
        hidden = out_gate * F.tanh(cell)

        # pack output
        new_state = (hidden, cell)
        output = hidden
        return output, new_state
