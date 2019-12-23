from utils import recursive_to_device
import copy
from utils import NormalizedAdder, NormalizedAdderList


def scatter(data, devices):
    '''
    Scatters data onto different devices.
    :param data: Data is a list of list. First dimension of list is time and second dimension is the batch items.
    :param devices: List of devices ids
    :return:
    '''

    ntimesteps = len(data)
    batch_size = len(data[0])

    minibatch_size = batch_size // len(devices)
    scat = []
    dev_id = 0
    for i in range(0, batch_size, minibatch_size):
        mini_data = [[recursive_to_device(data[t][b], devices[dev_id]) for b in range(i, min(i + minibatch_size, batch_size))] for t in range(ntimesteps)]
        scat += [mini_data]
        dev_id += 1
    return tuple(scat)


# copy state dict
def copy_state(old_state, new_state):
    assert len(old_state.size()) == len(new_state.size())
    sizes = []
    for i in range(len(old_state.size())):
        sizes += [min(old_state.size(i), new_state.size(i))]
    if len(sizes) == 1:
        new_state[:sizes[0]] = old_state[:sizes[0]]
    elif len(sizes) == 2:
        new_state[:sizes[0],:sizes[1]] = old_state[:sizes[0],:sizes[1]]
    elif len(sizes) == 3:
        new_state[:sizes[0],:sizes[1],:sizes[2]] = old_state[:sizes[0],:sizes[1],:sizes[2]]
    elif len(sizes) == 4:
        new_state[:sizes[0],:sizes[1],:sizes[2],:sizes[3]] = old_state[:sizes[0],:sizes[1],:sizes[2],:sizes[3]]
    elif len(sizes) == 5:
        new_state[:sizes[0],:sizes[1],:sizes[2],:sizes[3],:sizes[4]] = old_state[:sizes[0],:sizes[1],:sizes[2],:sizes[3],:sizes[4]]
    else:
        raise NotImplementedError
    return new_state


def collect_scattered_outputs(outputs, device):
    response = {}

    # accumulate responses
    for dev_id, output in enumerate(outputs):
        for type in output:
            if type == "info_scalar":
                if type not in response:
                    response[type] = {}
                for scalar in output[type]:
                    if scalar not in response[type]:
                        response[type][scalar] = NormalizedAdder(0)
                    response[type][scalar].append(output[type][scalar])
            if type == "info_temporal":
                if type not in response:
                    response[type] = {}
                for temporal_entry in output[type]:
                    if temporal_entry not in response[type]:
                        response[type][temporal_entry] = NormalizedAdderList(0, len(output[type][temporal_entry]))
                    response[type][temporal_entry].append_list(output[type][temporal_entry])
            if type == "__loss__":
                if type not in response:
                    response[type] = NormalizedAdder(recursive_to_device(output[type].new_zeros(1), device))
                response[type].append(recursive_to_device(output[type], device))
            if type == "__mu_Y__":
                if type not in response:
                    response[type] = {}
                for category in output[type]:
                    if category not in response[type]:
                        response[type][category] = {}
                    for t in output[type][category]:
                        if t not in response[type][category]:
                            response[type][category][t] = {}
                        existing_keys = list(response[type][category][t])
                        b_init = (max(existing_keys) + 1) if len(existing_keys) > 0 else 0
                        for _b in output[type][category][t]:
                            response[type][category][t][b_init+_b] = recursive_to_device(output[type][category][t][_b], device)

    # compile responses
    for type in response:
        if type == "info_scalar":
            for scalar in response[type]:
                response[type][scalar] = response[type][scalar].mean()
        if type == "info_temporal":
            for temporal_entry in response[type]:
                response[type][temporal_entry] = response[type][temporal_entry].mean_list()
        if type == "__loss__":
            response[type] = response[type].mean()

    return response

class NormalizedDictAdder:
    def __init__(self, initial):
        self.adder = {}
        self.initial = initial
    
    def append(self, dictionary):
        for type in dictionary:
            if type == "info_scalar":
                if type not in self.adder:
                    self.adder[type] = {}
                for scalar in dictionary[type]:
                    if scalar not in self.adder[type]:
                        self.adder[type][scalar] = NormalizedAdder(copy.deepcopy(self.initial))
                    self.adder[type][scalar].append(dictionary[type][scalar])
            if type == "info_temporal":
                if type not in self.adder:
                    self.adder[type] = {}
                for temporal_entry in dictionary[type]:
                    if temporal_entry not in self.adder[type]:
                        self.adder[type][temporal_entry] = NormalizedAdderList(copy.deepcopy(self.initial), len(dictionary[type][temporal_entry]))
                    self.adder[type][temporal_entry].append_list(dictionary[type][temporal_entry])

    def mean(self):
        response = copy.deepcopy(self.adder)
        for type in response:
            if type == "info_scalar":
                for scalar in response[type]:
                    response[type][scalar] = response[type][scalar].mean()
            if type == "info_temporal":
                for temporal_entry in response[type]:
                    response[type][temporal_entry] = response[type][temporal_entry].mean_list()
        return response