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


class Subvolume3DcnnModel(torch.nn.Module):
    def __init__(self, drop_rate, subvolume_shape, pad_to_shape,
                 batch_norm_momentum, no_batch_norm, filters, output_neurons):
        super().__init__()

        if pad_to_shape is not None:
            input_shape = pad_to_shape
        else:
            input_shape = subvolume_shape

        self._batch_norm = not no_batch_norm

        paddings = [1, 1, 1, 1]
        kernel_sizes = [3, 3, 3, 3]
        strides = [1, 2, 2, 2]

        self.conv1 = torch.nn.Conv3d(in_channels=1, out_channels=filters[0],
                                     kernel_size=kernel_sizes[0], stride=strides[0], padding=paddings[0])
        torch.nn.init.xavier_uniform(self.conv1.weight)
        torch.nn.init.zeros_(self.conv1.bias)
        self.batch_norm1 = torch.nn.BatchNorm3d(num_features=filters[0], momentum=batch_norm_momentum)
        shape = conv_output_shape(input_shape, kernel_sizes[0], strides[0], paddings[0])

        self.conv2 = torch.nn.Conv3d(in_channels=filters[0], out_channels=filters[1],
                                     kernel_size=kernel_sizes[1], stride=strides[1], padding=paddings[1])
        torch.nn.init.xavier_uniform(self.conv2.weight)
        torch.nn.init.zeros_(self.conv2.bias)
        self.batch_norm2 = torch.nn.BatchNorm3d(num_features=filters[1], momentum=batch_norm_momentum)
        shape = conv_output_shape(shape, kernel_sizes[1], strides[1], paddings[1])

        self.conv3 = torch.nn.Conv3d(in_channels=filters[1], out_channels=filters[2],
                                     kernel_size=kernel_sizes[2], stride=strides[2], padding=paddings[2])
        torch.nn.init.xavier_uniform(self.conv3.weight)
        torch.nn.init.zeros_(self.conv3.bias)
        self.batch_norm3 = torch.nn.BatchNorm3d(num_features=filters[2], momentum=batch_norm_momentum)
        shape = conv_output_shape(shape, kernel_sizes[2], strides[2], paddings[2])

        self.conv4 = torch.nn.Conv3d(in_channels=filters[2], out_channels=filters[3],
                                     kernel_size=kernel_sizes[3], stride=strides[3], padding=paddings[3])
        torch.nn.init.xavier_uniform(self.conv4.weight)
        torch.nn.init.zeros_(self.conv4.bias)
        self.batch_norm4 = torch.nn.BatchNorm3d(num_features=filters[3], momentum=batch_norm_momentum)
        shape = conv_output_shape(shape, kernel_sizes[3], strides[3], paddings[3])

        self.fc = torch.nn.Linear(filters[3] * np.prod(shape), output_neurons)
        self.dropout = torch.nn.Dropout(p=drop_rate)

        self.relu = torch.nn.ReLU()
        self.flatten = torch.nn.Flatten()

    def forward(self, x):
        y = self.conv1(x)
        y = self.relu(y)
        if self._batch_norm:
            y = self.batch_norm1(y)
        y = self.conv2(y)
        y = self.relu(y)
        if self._batch_norm:
            y = self.batch_norm2(y)
        y = self.conv3(y)
        y = self.relu(y)
        if self._batch_norm:
            y = self.batch_norm3(y)
        y = self.conv4(y)
        y = self.relu(y)
        if self._batch_norm:
            y = self.batch_norm4(y)
        y = self.flatten(y)
        y = self.fc(y)
        y = self.dropout(y)

        return y
