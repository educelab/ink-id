import math
from typing import Union

import numpy as np
import torch


def conv_output_shape(input_shape, kernel_size: Union[int, tuple], stride: Union[int, tuple],
                      padding: Union[int, tuple], dilation: Union[int, tuple] = 1):
    dim = len(input_shape)
    # Accept either ints or tuples for these parameters. If int, then convert into tuple (same value all for all dims).
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size,) * dim
    if isinstance(stride, int):
        stride = (stride,) * dim
    if isinstance(padding, int):
        padding = (padding,) * dim
    if isinstance(dilation, int):
        dilation = (dilation,) * dim
    # https://pytorch.org/docs/stable/nn.html#torch.nn.Conv3d See "Shape:" section.
    return tuple(math.floor((input_shape[d] + 2 * padding[d] - dilation[d] * (kernel_size[d] - 1) - 1) / stride[d] + 1)
                 for d in range(dim))


class Subvolume3DcnnEncoder(torch.nn.Module):
    def __init__(self, subvolume_shape,
                 batch_norm_momentum, no_batch_norm, filters, in_channels):
        super().__init__()

        input_shape = subvolume_shape

        self._batch_norm = not no_batch_norm
        self._in_channels = in_channels

        paddings = [1, 1, 1, 1]
        kernel_sizes = [3, 3, 3, 3]
        strides = [1, 2, 2, 2]

        self.conv1 = torch.nn.Conv3d(in_channels=in_channels, out_channels=filters[0],
                                     kernel_size=kernel_sizes[0], stride=strides[0], padding=paddings[0])
        torch.nn.init.xavier_uniform_(self.conv1.weight)
        torch.nn.init.zeros_(self.conv1.bias)
        self.batch_norm1 = torch.nn.BatchNorm3d(num_features=filters[0], momentum=batch_norm_momentum)
        shape = conv_output_shape(input_shape, kernel_sizes[0], strides[0], paddings[0])

        self.conv2 = torch.nn.Conv3d(in_channels=filters[0], out_channels=filters[1],
                                     kernel_size=kernel_sizes[1], stride=strides[1], padding=paddings[1])
        torch.nn.init.xavier_uniform_(self.conv2.weight)
        torch.nn.init.zeros_(self.conv2.bias)
        self.batch_norm2 = torch.nn.BatchNorm3d(num_features=filters[1], momentum=batch_norm_momentum)
        shape = conv_output_shape(shape, kernel_sizes[1], strides[1], paddings[1])

        self.conv3 = torch.nn.Conv3d(in_channels=filters[1], out_channels=filters[2],
                                     kernel_size=kernel_sizes[2], stride=strides[2], padding=paddings[2])
        torch.nn.init.xavier_uniform_(self.conv3.weight)
        torch.nn.init.zeros_(self.conv3.bias)
        self.batch_norm3 = torch.nn.BatchNorm3d(num_features=filters[2], momentum=batch_norm_momentum)
        shape = conv_output_shape(shape, kernel_sizes[2], strides[2], paddings[2])

        self.conv4 = torch.nn.Conv3d(in_channels=filters[2], out_channels=filters[3],
                                     kernel_size=kernel_sizes[3], stride=strides[3], padding=paddings[3])
        torch.nn.init.xavier_uniform_(self.conv4.weight)
        torch.nn.init.zeros_(self.conv4.bias)
        self.batch_norm4 = torch.nn.BatchNorm3d(num_features=filters[3], momentum=batch_norm_momentum)
        shape = conv_output_shape(shape, kernel_sizes[3], strides[3], paddings[3])
        self.output_shape = (filters[3],) + shape

    def forward(self, x):
        if self._in_channels > 1:
            x = torch.squeeze(x)
        y = self.conv1(x)
        y = torch.nn.functional.relu(y)
        if self._batch_norm:
            y = self.batch_norm1(y)
        y = self.conv2(y)
        y = torch.nn.functional.relu(y)
        if self._batch_norm:
            y = self.batch_norm2(y)
        y = self.conv3(y)
        y = torch.nn.functional.relu(y)
        if self._batch_norm:
            y = self.batch_norm3(y)
        y = self.conv4(y)
        y = torch.nn.functional.relu(y)
        if self._batch_norm:
            y = self.batch_norm4(y)

        return y


class Subvolume3DcnnDecoder(torch.nn.Module):
    def __init__(self, batch_norm_momentum, no_batch_norm, filters, in_channels):
        super().__init__()

        self._batch_norm = not no_batch_norm
        self._in_channels = in_channels

        paddings = [1, 1, 1, 1]
        kernel_sizes = [3, 3, 3, 3]
        strides = [1, 2, 2, 2]

        self.trans_conv1 = torch.nn.ConvTranspose3d(
            in_channels=filters[3], out_channels=filters[2], output_padding=1,
            kernel_size=kernel_sizes[3], stride=strides[3], padding=paddings[3]
        )
        torch.nn.init.xavier_uniform_(self.trans_conv1.weight)
        torch.nn.init.zeros_(self.trans_conv1.bias)
        self.batch_norm1 = torch.nn.BatchNorm3d(num_features=filters[2], momentum=batch_norm_momentum)

        self.trans_conv2 = torch.nn.ConvTranspose3d(
            in_channels=filters[2], out_channels=filters[1], output_padding=1,
            kernel_size=kernel_sizes[2], stride=strides[2], padding=paddings[2]
        )
        torch.nn.init.xavier_uniform_(self.trans_conv2.weight)
        torch.nn.init.zeros_(self.trans_conv2.bias)
        self.batch_norm2 = torch.nn.BatchNorm3d(num_features=filters[1], momentum=batch_norm_momentum)

        self.trans_conv3 = torch.nn.ConvTranspose3d(
            in_channels=filters[1], out_channels=filters[0], output_padding=1,
            kernel_size=kernel_sizes[1], stride=strides[1], padding=paddings[1]
        )
        torch.nn.init.xavier_uniform_(self.trans_conv3.weight)
        torch.nn.init.zeros_(self.trans_conv3.bias)
        self.batch_norm3 = torch.nn.BatchNorm3d(num_features=filters[0], momentum=batch_norm_momentum)

        self.trans_conv4 = torch.nn.ConvTranspose3d(
            in_channels=filters[0], out_channels=in_channels, output_padding=0,
            kernel_size=kernel_sizes[0], stride=strides[0], padding=paddings[0]
        )
        torch.nn.init.xavier_uniform_(self.trans_conv4.weight)
        torch.nn.init.zeros_(self.trans_conv4.bias)
        self.batch_norm4 = torch.nn.BatchNorm3d(num_features=in_channels, momentum=batch_norm_momentum)

    def forward(self, x):
        y = self.trans_conv1(x)
        y = torch.nn.functional.relu(y)
        if self._batch_norm:
            y = self.batch_norm1(y)
        y = self.trans_conv2(y)
        y = torch.nn.functional.relu(y)
        if self._batch_norm:
            y = self.batch_norm2(y)
        y = self.trans_conv3(y)
        y = torch.nn.functional.relu(y)
        if self._batch_norm:
            y = self.batch_norm3(y)
        y = self.trans_conv4(y)
        y = torch.nn.functional.relu(y)
        if self._batch_norm:
            y = self.batch_norm4(y)

        return y


class LinearInkDecoder(torch.nn.Module):
    def __init__(self, drop_rate, input_shape, output_neurons):
        super().__init__()

        self.fc = torch.nn.Linear(int(np.prod(input_shape)), output_neurons)
        self.dropout = torch.nn.Dropout(p=drop_rate)

        self.flatten = torch.nn.Flatten()

    def forward(self, x):
        y = self.flatten(x)
        y = self.fc(y)
        y = self.dropout(y)
        # Add some dimensions to match the dimensionality of label which is always 2D even if shape is (1, 1)
        y = torch.unsqueeze(y, 2)
        y = torch.unsqueeze(y, 3)
        return y  # (N, C, H, W)


class ConvolutionalInkDecoder(torch.nn.Module):
    def __init__(self, filters, output_channels):
        super().__init__()

        paddings = [1, 1, 1, 1]
        out_paddings = [0, 1, 1, 1]
        kernel_sizes = [3, 3, 3, 3]
        strides = [1, 2, 2, 2]

        self.leakyReLU = torch.nn.LeakyReLU()

        self.convtranspose1 = torch.nn.ConvTranspose2d(in_channels=filters[3], out_channels=filters[2],
                                                       kernel_size=kernel_sizes[3], stride=strides[3],
                                                       padding=paddings[3], output_padding=out_paddings[3])
        self.convtranspose2 = torch.nn.ConvTranspose2d(in_channels=filters[2], out_channels=filters[1],
                                                       kernel_size=kernel_sizes[2], stride=strides[2],
                                                       padding=paddings[2], output_padding=out_paddings[2])
        self.convtranspose3 = torch.nn.ConvTranspose2d(in_channels=filters[1], out_channels=filters[0],
                                                       kernel_size=kernel_sizes[1], stride=strides[1],
                                                       padding=paddings[1], output_padding=out_paddings[1])
        self.convtranspose4 = torch.nn.ConvTranspose2d(in_channels=filters[0], out_channels=output_channels,
                                                       kernel_size=kernel_sizes[0], stride=strides[0],
                                                       padding=paddings[0], output_padding=out_paddings[0])

    def forward(self, x):
        # Sum collapses from (B, C, D, H, W) down to (B, C, H, W)
        y = torch.sum(x, dim=2)
        # Transpose convolutions back up to the subvolume size (in 2D)
        y = self.convtranspose1(y)
        y = self.leakyReLU(y)
        y = self.convtranspose2(y)
        y = self.leakyReLU(y)
        y = self.convtranspose3(y)
        y = self.leakyReLU(y)
        y = self.convtranspose4(y)
        y = self.leakyReLU(y)
        return y


class Subvolume3DUNet(torch.nn.Module):
    def __init__(self,
                 subvolume_shape,
                 bn_momentum,
                 starting_channels,
                 in_channels,
                 decode=True):
        super().__init__()

        # Sanity check that our starting_channels is divisible by 2 since we
        # need to double and halve this parameter repeatedly
        if starting_channels % 2 != 0:
            raise ValueError('starting_channels must be divisible by 2')

        input_shape = subvolume_shape

        pool_kernel_size = 2
        pool_stride = 2
        pool_padding = 0  # Not listed in paper

        upconv_kernel_size = 2
        upconv_stride = 2
        upconv_padding = 0  # Not listed in paper

        final_conv_kernel_size = 1
        final_conv_stride = 1  # Not listed in paper, but stride 1 makes size work out
        final_conv_padding = 0  # Not listed in paper

        channels = starting_channels

        self._in_channels = in_channels
        self._encoder_modules = torch.nn.ModuleList()
        self._decoder_modules = torch.nn.ModuleList()
        self._decode = decode
        # The output shape depends on whether we run the decoder or not.
        out_channels = in_channels if decode else starting_channels * 16
        output_shape = [out_channels]
        output_shape.extend(input_shape if decode else map(lambda x: x / 8, input_shape))
        self.output_shape = tuple(output_shape)

        # Build the left side of the "U" shape plus bottom (encoder)
        for i in range(4):
            if i > 0:
                self._encoder_modules.append(
                    torch.nn.MaxPool3d(kernel_size=pool_kernel_size,
                                       stride=pool_stride,
                                       padding=pool_padding))
            self._encoder_modules.append(
                self._ConvBlock(in_channels=channels if i > 0 else in_channels,
                                out_channels=channels,
                                bn_momentum=bn_momentum))
            self._encoder_modules.append(
                self._ConvBlock(in_channels=channels,
                                out_channels=channels * 2,
                                bn_momentum=bn_momentum))
            channels *= 2

        # Build the right side of the "U" shape (decoder)
        for i in range(3):
            self._decoder_modules.append(
                torch.nn.ConvTranspose3d(in_channels=channels,
                                         out_channels=channels,
                                         kernel_size=upconv_kernel_size,
                                         stride=upconv_stride,
                                         padding=upconv_padding))
            channels //= 2
            self._decoder_modules.append(
                self._ConvBlock(in_channels=channels * 3,
                                out_channels=channels,
                                bn_momentum=bn_momentum))

            self._decoder_modules.append(
                self._ConvBlock(in_channels=channels,
                                out_channels=channels,
                                bn_momentum=bn_momentum))

        # Final decoder simple convolution (decoder)
        self._decoder_modules.append(
            torch.nn.Conv3d(in_channels=channels,
                            out_channels=out_channels,
                            kernel_size=final_conv_kernel_size,
                            stride=final_conv_stride,
                            padding=final_conv_padding))

    def forward(self, x):
        if self._in_channels > 1:
            x = torch.squeeze(x)

        # 0-indexed layers where shortcut/concatenation lines should appear on
        # both the encoder and decoder sides of the network.
        encoder_shortcut_layers = (1, 4, 7)
        decoder_shortcut_layers = (0, 3, 6)

        concat_lines = []

        for i, layer in enumerate(self._encoder_modules):
            x = layer(x)
            if i in encoder_shortcut_layers:
                concat_lines.append(x)

        # If we're not supposed to run the decoder section, just return now.
        if not self._decode:
            return x

        for i, layer in enumerate(self._decoder_modules):
            x = layer(x)
            if i in decoder_shortcut_layers:
                x = torch.cat((concat_lines.pop(), x), dim=1)

        return x

    class _ConvBlock(torch.nn.Module):
        def __init__(self,
                     in_channels,
                     out_channels,
                     bn_momentum,
                     kernel_size=3,
                     stride=1,
                     padding=1):
            super().__init__()
            # Padding needs to be 1 or our subvolume will shrink by 2 in each
            # dimension upon applying the Conv3d operation
            self.conv = torch.nn.Conv3d(in_channels=in_channels,
                                        out_channels=out_channels,
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        padding=padding)
            self.bn = torch.nn.BatchNorm3d(num_features=out_channels,
                                           momentum=bn_momentum)

        def forward(self, x):
            x = self.conv(x)
            x = self.bn(x)
            x = torch.nn.functional.relu(x)
            return x


class Autoencoder(torch.nn.Module):
    def __init__(self, subvolume_shape, batch_norm_momentum, no_batch_norm, filters):
        super().__init__()
        self.encoder = Subvolume3DcnnEncoder(subvolume_shape, batch_norm_momentum, no_batch_norm, filters,
                                             in_channels=1)
        self.decoder = Subvolume3DcnnDecoder(batch_norm_momentum, no_batch_norm, filters, in_channels=1)
        self.labels = ['autoencoded']

    def forward(self, x):
        return {
            'autoencoded': self.decoder(self.encoder(x)),
        }


class AutoencoderAndInkClassifier(torch.nn.Module):
    def __init__(self, subvolume_shape, batch_norm_momentum, no_batch_norm, filters, drop_rate):
        super().__init__()
        self.encoder = Subvolume3DcnnEncoder(subvolume_shape, batch_norm_momentum, no_batch_norm, filters,
                                             in_channels=1)
        self.ink_decoder = LinearInkDecoder(drop_rate, self.encoder.output_shape, output_neurons=2)
        self.autoencoder_decoder = Subvolume3DcnnDecoder(batch_norm_momentum, no_batch_norm, filters, in_channels=1)
        self.labels = ['autoencoded', 'ink_classes']

    def forward(self, x):
        x = self.encoder(x)
        autoencoded = self.autoencoder_decoder(x)
        ink = self.ink_decoder(x)

        return {
            'autoencoded': autoencoded,
            'ink_classes': ink,
        }


class InkClassifier3DCNN(torch.nn.Module):
    def __init__(self, subvolume_shape, batch_norm_momentum, no_batch_norm, filters, drop_rate):
        super().__init__()
        self.encoder = Subvolume3DcnnEncoder(subvolume_shape, batch_norm_momentum, no_batch_norm, filters,
                                             in_channels=1)
        self.decoder = LinearInkDecoder(drop_rate, self.encoder.output_shape, output_neurons=2)
        self.labels = ['ink_classes']

    def forward(self, x):
        return {
            'ink_classes': self.decoder(self.encoder(x)),
        }


class InkClassifier3DUNet(torch.nn.Module):
    def __init__(self, subvolume_shape_voxels, batch_norm_momentum, unet_starting_channels, in_channels, drop_rate):
        super().__init__()
        self.encoder = Subvolume3DUNet(subvolume_shape_voxels, batch_norm_momentum, unet_starting_channels, in_channels,
                                       decode=True)
        self.decoder = LinearInkDecoder(drop_rate, self.encoder.output_shape, output_neurons=2)
        self.labels = ['ink_classes']

    def forward(self, x):
        return {
            'ink_classes': self.decoder(self.encoder(x)),
        }


class InkClassifier3DUNetHalf(torch.nn.Module):
    def __init__(self, subvolume_shape_voxels, batch_norm_momentum, unet_starting_channels, in_channels, drop_rate):
        super().__init__()
        self.encoder = Subvolume3DUNet(subvolume_shape_voxels, batch_norm_momentum, unet_starting_channels, in_channels,
                                       decode=False)
        self.decoder = LinearInkDecoder(drop_rate, self.encoder.output_shape, output_neurons=2)
        self.labels = ['ink_classes']

    def forward(self, x):
        return {
            'ink_classes': self.decoder(self.encoder(x)),
        }


class RGB3DCNN(torch.nn.Module):
    def __init__(self, subvolume_shape, batch_norm_momentum, no_batch_norm, filters, drop_rate):
        super().__init__()
        self.encoder = Subvolume3DcnnEncoder(subvolume_shape, batch_norm_momentum, no_batch_norm, filters,
                                             in_channels=1)
        self.decoder = LinearInkDecoder(drop_rate, self.encoder.output_shape, output_neurons=3)
        self.labels = ['rgb_values']

    def forward(self, x):
        return {
            'rgb_values': self.decoder(self.encoder(x)),
        }


class InkClassifierCrossTaskVCTexture(torch.nn.Module):
    def __init__(self, subvolume_shape, batch_norm_momentum, no_batch_norm, filters, drop_rate):
        super().__init__()
        hidden_neurons = 20
        self.encoder = Subvolume3DcnnEncoder(subvolume_shape, batch_norm_momentum, no_batch_norm, filters,
                                             in_channels=1)
        self.decoder = LinearInkDecoder(drop_rate, self.encoder.output_shape, output_neurons=2)
        self.cross_task1 = torch.nn.Linear(2, hidden_neurons)
        self.cross_task2 = torch.nn.Linear(hidden_neurons, 1)
        self.labels = ['ink_classes', 'volcart_texture']

    def forward(self, x):
        x = self.encoder(x)
        ink = self.decoder(x)
        permuted = ink.permute(0, 2, 3, 1)  # From (N, C, H, W) to (N, H, W, C)
        texture = self.cross_task2(self.cross_task1(permuted))

        return {
            'ink_classes': ink,
            'volcart_texture': texture,
        }
