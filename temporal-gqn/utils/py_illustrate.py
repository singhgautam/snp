import json

from PIL import Image, ImageDraw, ImageFont
import PIL
import torch
import torchvision
import numpy as np
from torchvision import transforms

## LIBRARY OF COMPONENT HANDLERS

def make_tensor__text(
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

def make_tensor__truncated_imagegrid(
        input,
        img_size,
        grid_size,
        pad_value,
        padding,
        nchannels=3,
        blank_cell_color=(1.0, 1.0, 1.0),
    ):

    grid_h, grid_w = grid_size

    if len(tuple(img_size)) == 1:
        img_width, img_height = img_size, img_size
    else:
        img_width, img_height = img_size

    # blank image
    blank_image = torch.Tensor(blank_cell_color[:nchannels]).unsqueeze(-1).unsqueeze(-1)
    blank_image = blank_image.repeat(1, img_width, img_height)

    # remove None's from input
    for i in range(len(input)):
        for j in range(len(input[i])):
            if input[i][j] is None:
                input[i][j] = blank_image

    # truncate
    input_list = []
    for i in range(grid_h):
        input_row = []
        if i < len(input):
            input_row = input[i]
        input_row += [blank_image]*grid_w
        input_list += input_row[:grid_w]

    # unsqueeze and concatenate
    input_list = [tensor.unsqueeze(0) for tensor in input_list]
    for i in range(len(input_list)):
        if input_list[i].size(-3) == 1:
            input_list[i] = input_list[i].repeat(1,3,1,1)
    input = torch.cat(input_list, dim=0)
    output = torchvision.utils.make_grid(input, nrow=grid_w, normalize=True, scale_each=False, range=(0, 1.0), pad_value=pad_value, padding=padding)

    if grid_h == 1 and grid_w == 1:
        output = torch.nn.functional.pad(output, pad=[padding]*4, value=pad_value)

    return output



## LIBRARY OF COMPOUNDING LAYOUT

class Compounder:
    def __init__(self, component_tensors, spacing=2, h_justification="center", v_justification="center"):
        self.component_tensors = component_tensors
        self.spacing = int(spacing)
        self.h_justification = h_justification
        self.v_justification = v_justification

    def compound_array_as_columns(self, array):
        tensor_array = []
        for item in array:
            if type(item) == dict:
                tensor = self.compound_object(item)
            elif type(item) == str:
                tensor = self.component_tensors[item]
            tensor_array += [tensor]

        max_height = int(max([tensor.size(1) for tensor in tensor_array]))
        total_width = int(sum([tensor.size(2) for tensor in tensor_array]) + self.spacing*(len(tensor_array)-1))
        canvas = torch.ones(3,max_height, total_width)

        cursor = 0
        for tensor in tensor_array:
            if self.v_justification == "center":
                h_pos = int((max_height - tensor.size(1))/2)
            elif self.v_justification == "top":
                h_pos = 0
            canvas[:,h_pos:h_pos+tensor.size(1),cursor:cursor+tensor.size(2)] = tensor
            cursor += int(tensor.size(2) + self.spacing)
        return canvas

    def compound_array_as_rows(self, array):
        tensor_array = []
        for item in array:
            if type(item) == dict:
                tensor = self.compound_object(item)
            elif type(item) == str:
                tensor = self.component_tensors[item]
            tensor_array += [tensor]

        max_width = int(max([tensor.size(2) for tensor in tensor_array]))
        total_height = int(sum([tensor.size(1) for tensor in tensor_array]) + self.spacing*(len(tensor_array)-1))
        canvas = torch.ones(3,total_height, max_width)

        cursor = 0
        for tensor in tensor_array:
            if self.h_justification == "center":
                w_pos = int((max_width - tensor.size(2))/2)
            elif self.h_justification == "left":
                w_pos = 0
            canvas[:,cursor:cursor+tensor.size(1),w_pos:w_pos+tensor.size(2)] = tensor
            cursor += int(tensor.size(1) + self.spacing)
        return canvas

    def compound_object(self, object):
        axis = list(object)[0]
        array = object[axis]
        if axis == "rows":
            return self.compound_array_as_rows(array)
        elif axis == "columns":
            return self.compound_array_as_columns(array)
        else:
            raise NotImplementedError

## CALLER FUNCTIONS

def py_illustrate(file, data):
    # load template
    with open(file, 'r') as f:
        template = json.load(f)

    # produce component tensors
    component_tensors = {}
    for component in template["components"]:
        _label = component["label"]
        _tensor = process_component(component["type"], data[_label] if _label in data else None, component["details"])
        component_tensors[_label] = _tensor

    # arrange component tensors
    compounder = Compounder(component_tensors)
    final_image = compounder.compound_object(template["placements"])

    return final_image

def process_component(type, data, details):
    if type == "text":
        return make_tensor__text(
            text=data,
            img_size=tuple(details["img_size"]),
            bg_color=details["bg_color"],
            font_size=details["font_size"],
            font_file=details["font_file"],
            font_color=tuple(details["font_color"]),
            text_placement=tuple(details["text_placement"]),
        )
    if type == "static_text":
        return make_tensor__text(
            text=details["text"],
            img_size=tuple(details["img_size"]),
            bg_color=details["bg_color"],
            font_size=details["font_size"],
            font_file=details["font_file"],
            font_color=tuple(details["font_color"]),
            text_placement=tuple(details["text_placement"]),
        )
    if type == "truncated_imagegrid":
        return make_tensor__truncated_imagegrid(
            input=data,
            img_size=tuple(details["img_size"]),
            grid_size=tuple(details["grid_size"]),
            pad_value=details["pad_value"],
            padding=details["padding"],
            nchannels=details["img_channels"] if 'img_channels' in details else 3,
            blank_cell_color=tuple(details["blank_cell_color"]) if 'blank_cell_color' in details else (1.0, 1.0, 1.0),
        )

    if type == "image":
        return data

    if type == "image_path":
        path = details["path"]
        img_h = details["img_size"]
        img_w = details["img_size"]

        pil_img = Image.open(path).convert('RGB')
        pil_img = pil_img.resize((img_w, img_h), PIL.Image.BILINEAR)
        image = np.array(pil_img)
        ret = torch.from_numpy(image / 255).permute(2, 0, 1).float()
        return ret

    if type == "truncated_imagepathgrid":
        img_h = details["img_size"]
        img_w = details["img_size"]
        for i in range(len(data)):
            for j in range(len(data[i])):
                path = data[i][j]
                pil_img = Image.open(path).convert('RGB')
                pil_img = pil_img.resize((img_w, img_h), PIL.Image.BILINEAR)
                image = np.array(pil_img)
                data[i][j] = torch.from_numpy(image / 255).permute(2, 0, 1).float()

        ret = make_tensor__truncated_imagegrid(
            input=data,
            img_size=tuple(details["img_size"]),
            grid_size=tuple(details["grid_size"]),
            pad_value=details["pad_value"],
            padding=details["padding"],
            nchannels=details["img_channels"] if 'img_channels' in details else 3,
            blank_cell_color=tuple(details["blank_cell_color"]) if 'blank_cell_color' in details else (1.0, 1.0, 1.0),
        )

    if type == "truncated_textgrid":
        for i in range(len(data)):
            for j in range(len(data[i])):
                text = data[i][j]
                data[i][j] = make_tensor__text(
                    text=text,
                    img_size=tuple(details["img_size"]),
                    bg_color=details["bg_color"],
                    font_size=details["font_size"],
                    font_file=details["font_file"],
                    font_color=tuple(details["font_color"]),
                    text_placement=tuple(details["text_placement"]),
                )

        ret = make_tensor__truncated_imagegrid(
            input=data,
            img_size=tuple(details["img_size"]),
            grid_size=tuple(details["grid_size"]),
            pad_value=details["pad_value"],
            padding=details["padding"],
            nchannels=details["img_channels"] if 'img_channels' in details else 3,
            blank_cell_color=tuple(details["blank_cell_color"]) if 'blank_cell_color' in details else (1.0, 1.0, 1.0),
        )

        return ret

class Illustrator:
    def __init__(self, template_file):
        with open(template_file, 'r') as f:
            self.template = json.load(f)

    def illustrate(self, data):
        # produce component tensors
        component_tensors = {}
        for component in self.template["components"]:
            _label = component["label"]
            _tensor = process_component(component["type"], data[_label] if _label in data else None, component["details"])
            component_tensors[_label] = _tensor

        # arrange component tensors
        compounder = Compounder(
            component_tensors,
            spacing=self.template["metadata"]["spacing"] if "spacing" in self.template["metadata"] else 2,
            h_justification=self.template["metadata"]["h_justification"] if "spacing" in self.template["metadata"] else "center",
            v_justification=self.template["metadata"]["v_justification"] if "spacing" in self.template["metadata"] else "center",
        )
        final_image = compounder.compound_object(self.template["placements"])

        return final_image