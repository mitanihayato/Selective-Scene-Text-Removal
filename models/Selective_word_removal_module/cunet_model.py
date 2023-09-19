import torch
import torch.nn as nn

from models.Selective_word_removal_module.FiLM_utils import FiLM_complex, FiLM_simple
from models.Selective_word_removal_module.control_models import dense_control_block, dense_control_model
from models.Selective_word_removal_module.unet_model import u_net_deconv_block, Conv2d_same, ConvTranspose2d_same


def get_func_by_name(activation_str):
    if activation_str == "leaky_relu":
        return nn.LeakyReLU
    elif activation_str == "relu":
        return nn.ReLU
    elif activation_str == "sigmoid":
        return nn.Sigmoid
    elif activation_str == "tanh":
        return nn.Tanh
    elif activation_str == "softmax":
        return nn.Softmax
    elif activation_str == "identity":
        return nn.Identity
    else:
        return None


class u_net_conv_block(nn.Module):
    def __init__(self, conv_layer, bn_layer, FiLM_layer, activation):
        super(u_net_conv_block, self).__init__()
        self.bn_layer = bn_layer
        self.conv_layer = conv_layer
        self.FiLM_layer = FiLM_layer
        self.activation = activation
        self.in_channels = self.conv_layer.conv.in_channels
        self.out_channels = self.conv_layer.conv.out_channels

    def forward(self, x, gamma, beta):
        x = self.bn_layer(self.conv_layer(x))
        x = self.FiLM_layer(x, gamma, beta)
        return self.activation(x)


class CUNET(nn.Module):

    @staticmethod
    def get_arg_keys():
        return ['n_layers',
                'input_channels',
                'filters_layer_1',
                'kernel_size',
                'stride',
                'film_type',
                'control_type',
                'encoder_activation',
                'decoder_activation',
                'last_activation',
                'control_input_dim',
                'control_n_layer']

    def __init__(self,
                 n_layers,
                 input_channels,
                 filters_layer_1,
                 kernel_size=(5, 5),
                 stride=(2, 2),
                 film_type='simple',
                 control_type='dense',
                 encoder_activation="leaky_relu",
                 decoder_activation="relu",
                 last_activation="sigmoid",
                 control_input_dim=4,
                 control_n_layer=4
                 ):
        '''
            n_layers : number of encoder block
            input_chanels : input channels
            filter_layer : number of channels in the output image of the first encoder block
            kernel_size : kernel size
            stride : stride
            film_tipe : type of FiLM layer (simple or complex)
            control_type : dense or conv currently, only dense is supported
            control_input_dim : input dimension of control model
            control_n_layer : number of layers in control model
        '''

        self.encoder_activation = encoder_activation
        self.decoder_activation = decoder_activation
        self.last_activation = last_activation

        encoder_activation = get_func_by_name(self.encoder_activation)
        decoder_activation = get_func_by_name(self.decoder_activation)
        last_activation = get_func_by_name(self.last_activation)

        super(CUNET, self).__init__()

        self.input_control_dims = control_input_dim
        self.n_layers = n_layers
        self.input_channels = input_channels
        self.filters_layer_1 = filters_layer_1
        self.kernel_size = kernel_size
        self.stride = stride
        self.film_type = film_type
        self.control_type = control_type
        self.control_n_layer = control_n_layer


        # Encoder
        encoders = []
        for i in range(n_layers):
            output_channels = filters_layer_1 * (2 ** i)
            encoders.append(
                u_net_conv_block(
                    conv_layer=Conv2d_same(input_channels, output_channels, kernel_size, stride),
                    bn_layer=nn.BatchNorm2d(output_channels),
                    FiLM_layer=FiLM_simple if film_type == "simple" else FiLM_complex,
                    activation=encoder_activation()
                )
            )
            input_channels = output_channels
        self.encoders = nn.ModuleList(encoders)

        # Decoder
        decoders = []
        for i in range(n_layers):
            # parameters each decoder layer
            is_final_block = i == n_layers - 1  # the las layer is different
            # not dropout in the first block and the last two encoder blocks
            dropout = not (i == 0 or i == n_layers - 1 or i == n_layers - 2)
            # for getting the number of filters
            encoder_layer = self.encoders[n_layers - i - 1]
            skip = i > 0  # not skip in the first encoder block

            input_channels = encoder_layer.out_channels
            if skip:
                input_channels *= 2

            if is_final_block:
                output_channels = self.input_channels
                activation = last_activation
            else:
                output_channels = encoder_layer.in_channels
                activation = decoder_activation

            decoders.append(
                u_net_deconv_block(
                    deconv_layer=ConvTranspose2d_same(input_channels, output_channels, kernel_size, stride),
                    bn_layer=nn.BatchNorm2d(output_channels),
                    activation=activation(),
                    dropout=dropout
                )
            )

            self.decoders = nn.ModuleList(decoders)

        # Control Mechanism
        if film_type == "simple":
            control_output_dim = n_conditions = n_layers
            split = lambda tensor: [tensor[..., i] for i in range(n_layers)]
        else:
            output_channel_array = [encoder.conv_layer.conv.out_channels for encoder in self.encoders]
            control_output_dim = n_conditions = sum(output_channel_array)

            start_idx_per_layer = [sum(output_channel_array[:i]) for i in range(len(output_channel_array))]
            end_idx_per_layer = [sum(output_channel_array[:i + 1]) for i in range(len(output_channel_array))]

            split = lambda tensor: [tensor[..., start:end] for start, end in
                                    zip(start_idx_per_layer, end_idx_per_layer)]
        if control_type == "dense":
            self.condition_generator = dense_control_model(
                dense_control_block(control_input_dim, control_n_layer),
                control_output_dim,
                split
            )
        else:
            raise NotImplementedError

    def forward(self, input_spec, input_condition):

        gammas, betas = self.condition_generator(input_condition)

        x = input_spec

        # Encoding Phase
        encoder_outputs = []
        for encoder, gamma, beta in zip(self.encoders, gammas, betas):
            encoder_outputs.append(encoder(x, gamma, beta))  # TODO
            x = encoder_outputs[-1]

        # Decoding Phase
        x = self.decoders[0](x)
        for decoder, x_encoded in zip(self.decoders[1:], reversed(encoder_outputs[:-1])):
            x = decoder(torch.cat([x, x_encoded], dim=-3))

        return x