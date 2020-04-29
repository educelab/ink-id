import math
from typing import Union

import numpy as np
import torch


def conv_output_shape(input_shape, kernel_size: Union[int, tuple], stride: Union[int, tuple],
                      padding: Union[int, tuple], dilation: Union[int, tuple] = 1):
    dim = len(input_shape)
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

        self.conv1 = torch.nn.Conv3d(in_channels=1, out_channels=filters[0],
                                     kernel_size=3, stride=1, padding=1)
        self.batch_norm1 = torch.nn.BatchNorm3d(num_features=filters[0], momentum=batch_norm_momentum)
        shape = conv_output_shape(input_shape, 3, 1, 1)

        self.conv2 = torch.nn.Conv3d(in_channels=filters[0], out_channels=filters[1],
                                     kernel_size=3, stride=2, padding=1)
        self.batch_norm2 = torch.nn.BatchNorm3d(num_features=filters[1], momentum=batch_norm_momentum)
        shape = conv_output_shape(shape, 3, 2, 1)

        self.conv3 = torch.nn.Conv3d(in_channels=filters[1], out_channels=filters[2],
                                     kernel_size=3, stride=2, padding=1)
        self.batch_norm3 = torch.nn.BatchNorm3d(num_features=filters[2], momentum=batch_norm_momentum)
        shape = conv_output_shape(shape, 3, 2, 1)

        self.conv4 = torch.nn.Conv3d(in_channels=filters[2], out_channels=filters[3],
                                     kernel_size=3, stride=2, padding=1)
        self.batch_norm4 = torch.nn.BatchNorm3d(num_features=filters[3], momentum=batch_norm_momentum)
        shape = conv_output_shape(shape, 3, 2, 1)

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
