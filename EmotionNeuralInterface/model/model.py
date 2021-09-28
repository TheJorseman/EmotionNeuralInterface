import torch.nn as nn
from .encoder import CNN1D
from .stage_net import FullyConected
from math import prod

class SiameseNetwork(nn.Module):
    def __init__(self, window_size=512, conv_kernel=9, stride=5, dropout=0.3):
        super(SiameseNetwork, self).__init__()
        channels = (64, 64, 64, 16)
        # 64-64-64
        # 
        #mp_kernel1 = int(window_size - (conv_kernel - 1) - (window_size/2) + 1)
        #print("mp_kernel1", mp_kernel1)
        self.encoder1 = CNN1D(channels[0], kernel_size=5, mp_kernel=1)
        output1 = self.calculate_output(window_size, 5, stride, 2, 1) 
        self.encoder2 = CNN1D(channels[1], channels=channels[0], kernel_size=5, mp_kernel=2, mp_stride=2)
        output2 = self.calculate_output(output1, 3, stride, 2, 2) 
        self.encoder3 = CNN1D(channels[2], channels=channels[1], kernel_size=5, mp_kernel=2, mp_stride=2)
        output3 = self.calculate_output(output2, 3, stride, 5, 2) 
        #mp_kernel = self.get_kernel_mp(256, output3, 3, stride, 1)
        #self.encoder4 = CNN1D(channels[3], channels=channels[2], kernel_size=3, mp_kernel=mp_kernel)     
        #output4 = self.calculate_output(output3, 3, stride, 9, 1)  
        self.fc1 = nn.Linear(7936, 64)
        #self.fc2 = nn.Linear(dim[3], dim[4])
        self.normalization = nn.BatchNorm1d(64)
        #self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

    def get_kernel_mp(self, expected, input, conv_kernel, stride , mp_stride):
        conv_out = self.calculate_conv_output(input, conv_kernel, stride)
        mp_kernel = int(conv_out + 1 - (expected * mp_stride))
        print("Kernel MP = ", mp_kernel)
        return mp_kernel

    def calculate_conv_output(self, input, conv_kernel, stride):
        return int((input - conv_kernel + 1)/stride)
    
    def calculate_mp_output(self, conv_out, mp_kernel, mp_stride):
        return int((conv_out - mp_kernel + 1)/mp_stride)

    def calculate_output(self, input, conv_kernel, stride, mp_kernel, mp_stride):
        conv = self.calculate_conv_output(input,conv_kernel,stride)
        output = self.calculate_mp_output(conv, mp_kernel, mp_stride)
        print(output)
        return output

    def forward_once(self, x):
        # Forward pass 
        #import pdb;pdb.set_trace()
        output = self.encoder1(x.unsqueeze(1))
        #output = output.view(output.size()[0], -1)
        output = self.encoder2(output)
        output = self.encoder3(output)
        #output = self.encoder4(output)
        #output = output.view(output.size()[0], -1)
        #output = self.encoder3(output.unsqueeze(1))
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        output = self.relu(output)
        #output = self.fc2(output)
        #output = self.relu(output)
        output = self.normalization(output)       
        return self.dropout(output)

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