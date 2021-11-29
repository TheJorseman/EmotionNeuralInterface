import torch.nn as nn
from numpy import prod
import json
from .model import SiameseLinearNetwork
from .model import SiameseNetwork
from .stage_net import StageNet, FullyConected, Encoder
from .nedbert import NedBERT
from .encoder import CNN1D

def select_model(path, window_size, multichannel_len):
    """
    Select the model to use.
    """
    model_config = json.load(open(path, "r"))
    name = model_config["name"]
    if name == "siamese_stagenet":
        return StageNet(model_config, width=window_size,height=multichannel_len)
    elif name == "siamese_conv":
        return SiameseNetwork(model_config, window_size=window_size)
    elif name == "siamese_linear":
        return SiameseLinearNetwork(model_config, window_size=window_size)
    elif name == "nedbert":
        return NedBERT(model_config, sequence_lenght=window_size)
    raise Warning("No type model found")


class Classificator(nn.Module):
    def __init__(self, config_model, pretrained=False, window_size=512, multichannel_len=14, channels_in=1):
        super(Classificator, self).__init__()
        self.config_model = config_model
        self.multichannel_len = multichannel_len
        if not pretrained:
            self.pre_model = select_model(config_model["use_model"], window_size, multichannel_len)
        else:
            self.pre_model = pretrained
        embedding_dim = self.pre_model.layers[-1].linear.out_features
        layers, shapes = self.get_modules(config_model["layers"], (channels_in, embedding_dim))
        self.layers = nn.ModuleList(layers)
        self.shapes = shapes

    def get_linear_input_dim(self, values):
        return prod(values)

    def get_modules(self, layers, dim_tuple):
        modules = []
        shapes = [dim_tuple]
        for key in layers.keys():
            if layers[key]["output_dim"] == -1:
                layers[key]["output_dim"] = self.multichannel_len
            if "conv" in key:
                modules.append(CNN1D(layers[key], channels_in=shapes[-1][0], l_in=shapes[-1][1]))
            elif "linear" in key:
                modules.append(FullyConected(layers[key], self.get_linear_input_dim(shapes[-1])))
            elif "conv2d" in key:
                modules.append(Encoder(layers[key], channels_in=shapes[-1][0], h_in=shapes[-1][1], w_in=shapes[-1][2]))
            print(shapes)
            shapes.append(modules[-1].calculate_output_shape())
        return modules, shapes

    def forward_pretrained(self, input):
        output = input.squeeze()
        return self.pre_model.forward_once(output)

    def forward(self, input):
        # Forward pass 
        output = self.forward_pretrained(input)
        for layer in self.layers:
            output = layer(output)
        return output
