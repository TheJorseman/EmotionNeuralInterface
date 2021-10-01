import torch.nn as nn
from .encoder import CNN1D
from .stage_net import FullyConected
from numpy import prod

class SiameseNetwork(nn.Module):
    def __init__(self, value_dict, channels_in=1, window_size=512):
        super(SiameseNetwork, self).__init__()
        #channels = (64, 64, 64, 16)
        self.value_dict = value_dict
        layers, shapes = self.get_modules(value_dict["layers"], (channels_in, window_size))
        self.layers = nn.ModuleList(layers)
        self.shapes = shapes

    def get_linear_input_dim(self, values):
        return prod(values)

    def get_modules(self, layers, dim_tuple):
        modules = []
        shapes = [dim_tuple]
        for key in layers.keys():
            if "conv" in key:
                modules.append(CNN1D(layers[key], channels_in=shapes[-1][0], l_in=shapes[-1][1]))
            elif "linear" in key:
                modules.append(FullyConected(layers[key], self.get_linear_input_dim(shapes[-1])))
            print(shapes)
            shapes.append(modules[-1].calculate_output_shape())
        return modules, shapes

    def forward_once(self, x):
        # Forward pass 
        output = x.unsqueeze(1)
        for layer in self.layers:
            output = layer(output)
        return output

    def forward(self, input1, input2):
        # forward pass of input 1
        output1 = self.forward_once(input1)
        # forward pass of input 2
        output2 = self.forward_once(input2)
        return output1, output2


class SiameseLinearNetwork(nn.Module):
    def __init__(self, value_dict, window_size=512):
        super(SiameseLinearNetwork, self).__init__()
        layers, shapes = self.get_modules(value_dict["layers"], (window_size, 1))
        self.layers = nn.ModuleList(layers)
        self.shapes = shapes
        self.dropout = nn.Dropout(p=value_dict["dropout"])
        print(shapes)

    def get_linear_input_dim(self, values):
        return prod(values)

    def get_modules(self, layers, dim_tuple):
        modules = []
        shapes = [dim_tuple]
        for key in layers.keys():
            if "linear" in key:
                modules.append(FullyConected(layers[key], self.get_linear_input_dim(shapes[-1])))
            print(shapes)
            shapes.append(modules[-1].calculate_output_shape())
        return modules, shapes

    def forward_once(self, x):
        output = x
        for layer in self.layers:
            output = layer(output)
        return output

    def forward(self, input1, input2):
        # forward pass of input 1
        output1 = self.forward_once(input1)
        # forward pass of input 2
        output2 = self.forward_once(input2)
        return output1, output2