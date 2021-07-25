import torch.nn as nn
from torch import transpose, flatten, zeros, arange
import torch
import math

class NedBERT(nn.Module):
    def __init__(self, values, device=False):
        super(NedBERT,self).__init__()
        self.values = values
        self.device = self.device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.spatial_conv = SpatialConv(values["spatial_conv"])
        self.channels = values["spatial_conv"]["channels_out"]
        self.sequence_length = values["sequence_length"]
        t_val = values["transformer"]
        po = values["positional_encoding"]
        d_model = values["d_model"]
        self.pos_encoder = PositionalEncoding(d_model, po["dropout"])
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, 
                                                    nhead=t_val["nhead"], 
                                                    dim_feedforward=t_val["dim_feedforward"],
                                                    dropout=t_val["dropout"],
                                                    activation=t_val["activation"])
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=t_val["num_encoder_layers"])
        self.d_model = d_model
        #self.transformer = nn.Transformer(t_val["d_model"], t_val["nhead"], 
        #                                    num_encoder_layers=t_val["num_encoder_layers"], num_decoder_layers=t_val["num_decoder_layers"],
        #                                    dim_feedforward=t_val["dim_feedforward"], dropout=t_val["dropout"], activation=t_val["activation"])

    def generate_square_subsequent_mask(self):
        size1 = self.channels
        size2 = self.d_model
        mask = (torch.triu(torch.ones(size2, size2)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward_once(self, x, mask=None):
        import pdb;pdb.set_trace()
        output = self.spatial_conv(x.unsqueeze(1))
        output = output.squeeze()
        output = output.transpose(0, 2).transpose(1,2)
        output = self.pos_encoder(output)
        output = self.transformer_encoder(output).transpose(0,1).transpose(1,2)
        return output.mean(2)

    def forward(self, input1, input2):
        # forward pass of input 1
        #import pdb;pdb.set_trace()
        mask1 = self.generate_square_subsequent_mask().to(self.device)
        output1 = self.forward_once(input1)
        # forward pass of input 2
        mask2 = self.generate_square_subsequent_mask().to(self.device)
        output2 = self.forward_once(input2)
        #import pdb;pdb.set_trace()
        return output1, output2


class SpatialConv(nn.Module):
    def __init__(self, values_dict, channel_in=1):
        super(SpatialConv,self).__init__()
        kernel_conf = values_dict["kernel"]
        kernel_size = kernel_conf if type(kernel_conf) == type(int()) else tuple(kernel_conf)
        self.spatial_conv = nn.Conv2d(channel_in, values_dict["channels_out"], kernel_size)

    def forward(self, x):
        output = self.spatial_conv(x)
        return transpose(output, 1, 2)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = zeros(max_len, d_model)
        position = arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


