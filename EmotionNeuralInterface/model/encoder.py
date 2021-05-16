import torch.nn as nn
from torch import transpose

class CNN1D(nn.Module):
    """
    Este es el modulo que realiza una Convolución + Maxpool + dropout.
    """    
    def __init__(self, filters, channels=1, kernel_size=9, stride=1, mp_kernel=10, mp_stride=1, dropout=0.3, input_size=1024):
        super(CNN1D,self).__init__()
        self.conv1d = nn.Conv1d(channels, filters, kernel_size)
        #self.conv_output = int((input_size - (kernel_size-1) -1)/stride)
        #self.maxpool_output = int((self.conv_output - (mp_kernel-1) - 1)/mp_stride)
        self.gelu = nn.GELU()
        self.relu = nn.ReLU()
        self.normalization = nn.BatchNorm1d(filters)
        self.maxpool = nn.MaxPool1d(mp_kernel, stride=mp_stride)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        output = self.conv1d(x)
        output = self.relu(output)
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