from numpy import prod
import yaml

class StageNet():
    def __init__(self, value_dict, channels_in=1, height=14, width=512):
        _, shapes = self.get_modules(value_dict["layers"], (channels_in, height, width))
        print(shapes)

    def get_linear_input_dim(self, values):
        return prod(values)

    def get_modules(self, layers, dim_tuple):
        modules = []
        shapes = [dim_tuple]
        for key in layers.keys():
            if "conv" in key:
                modules.append(Encoder(layers[key], channels_in=shapes[-1][0], h_in=shapes[-1][1], w_in=shapes[-1][2]))
            elif "linear" in key:
                modules.append(FullyConected(layers[key], self.get_linear_input_dim(shapes[-1])))
            print(shapes)
            shapes.append(modules[-1].calculate_output_shape())
        return modules, shapes

class FullyConected():
    def __init__(self, config, input_dim):
        self.config = config

    def calculate_output_shape(self):
        return [self.config["output_dim"]]

class Encoder():
    def __init__(self, values_dict, channels_in=1, h_in=14, w_in=512):
        self.channel_in = channels_in
        self.h_in = h_in
        self.w_in = w_in
        self.values_dict = values_dict

    def get_kernel_from_dict(self, value):
        return value if type(value) == type(int()) else tuple(value)

    def calculate_conv_shape(self, h_in, w_in, value_dict):
        conv_kernel = self.get_kernel_from_dict(value_dict["kernel"])
        stride = self.get_kernel(value_dict["stride"])
        return (value_dict["channels_out"], self.dim_out(h_in, conv_kernel[0], stride[0]) , self.dim_out(w_in, conv_kernel[1], stride[1]))

    def dim_out(self, val_in, kernel, stride, padding=0, dilatation=1):
        return int(((val_in + 2 * padding - dilatation * (kernel -1) - 1)/stride) + 1)

    def calculate_mp_shape(self, conv_tuple, value_dict):
        #conv_kernel = self.get_kernel_from_dict(value_dict["kernel"])
        conv_kernel = value_dict["kernel"]
        stride = self.get_kernel(value_dict["stride"])
        return (conv_tuple[0], self.dim_out(conv_tuple[1], conv_kernel[0], stride[0]) , self.dim_out(conv_tuple[2],conv_kernel[1],stride[1]))  

    def calculate_output_shape(self):
        output = self.calculate_conv_shape(self.h_in, self.w_in, self.values_dict)
        return self.calculate_mp_shape(output, self.values_dict["maxpool"])

    def get_kernel(self, value):
        if isinstance(value, int):
            return (value,value)
        else:
            return tuple(value)

if __name__ == '__main__':
    file = 'config/models/stagenet.yaml'
    with open(file) as f:
        model_config = yaml.load(f, Loader=yaml.FullLoader)
        StageNet(model_config, width=256)


