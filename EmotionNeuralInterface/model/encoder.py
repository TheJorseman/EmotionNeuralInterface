import torch.nn as nn
from torch import transpose

class CNN1D(nn.Module):
    """
    Este es el modulo que realiza una Convolución + Maxpool + dropout.
    """    
    def __init__(self, values_dict, channels_in=1, l_in=512):
        super(CNN1D,self).__init__()
        self.channels_in = channels_in
        self.l_in = l_in
        self.values_dict = values_dict
        self.conv1d = nn.Conv1d(channels_in, values_dict['channels_out'], values_dict['kernel'], stride=values_dict['stride'])
        self.fn_act = self.get_activation_fn(values_dict['act_fn'])
        mp_vals = values_dict['maxpool']
        self.maxpool = nn.MaxPool1d(self.get_kernel_from_dict(mp_vals['kernel']), stride=self.get_kernel_from_dict(mp_vals['stride']))
        self.norm = bool(values_dict["batch_normalization"])
        self.batch_normalization = nn.BatchNorm1d(values_dict['channels_out'])
        self.dropout = nn.Dropout(p=values_dict['dropout'])

    def normalization(self, x):
        if self.norm:
            return self.batch_normalization(x)
        return x

    def get_kernel_from_dict(self, value):
        return value if type(value) == type(int()) else tuple(value)

    def get_activation_fn(self, value):
        if value == "relu":
            return nn.ReLU()
        elif value == "gelu":
            return nn.GELU()
        raise Warning("No supported activation function")

    def calculate_output(self, l_in, kernel_size, stride=1, padding=0, dilatation=1):
        return int(((l_in + 2*padding - dilatation*(kernel_size-1) - 1)/stride) + 1)

    def calculate_conv1d_output(self, values_dict):
        return (values_dict['channels_out'], self.calculate_output(self.l_in, values_dict['kernel'], stride=values_dict['stride']))

    def calculate_maxpool_output(self, values_dict, channels_out, l_in):
        return (channels_out, self.calculate_output(l_in, values_dict['kernel'], stride=values_dict['stride']))

    def calculate_output_shape(self):
        # Calculate Conv1D output
        output_shape = self.calculate_conv1d_output(self.values_dict)
        return self.calculate_maxpool_output(self.values_dict['maxpool'], output_shape[0], output_shape[1])

    def forward(self, x):
        output = self.conv1d(x)
        output = self.fn_act(output)
        output= self.normalization(output)
        output = self.maxpool(output)
        return self.dropout(output)

class CNN2D(nn.Module):
    """
    Este es el modulo que realiza una Convolución + Maxpool + dropout.
    """    
    def __init__(self, filters, channels=1, kernel_size=9, stride=1, mp_kernel=10, mp_stride=1, dropout=0.3):
        super(CNN2D,self).__init__()
        self.conv = nn.Conv2d(channels, filters, kernel_size)
        self.gelu = nn.GELU()
        self.relu = nn.ReLU()
        self.normalization = nn.BatchNorm2d(filters)
        self.maxpool = nn.MaxPool2d(mp_kernel, stride=mp_stride)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        output = self.conv(x)
        output = self.gelu(output)
        output= self.normalization(output)
        output = self.maxpool(output)
        return self.dropout(output)    