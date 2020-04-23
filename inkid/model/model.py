import torch


class Subvolume3DcnnModel(torch.nn.Module):
    def __init__(self, drop_rate, subvolume_shape, pad_to_shape,
                 batch_norm_momentum, no_batch_norm, filters, output_neurons):
        super().__init__()

        self._batch_norm = not no_batch_norm

        self.conv1 = torch.nn.Conv3d(in_channels=1, out_channels=filters[0],
                                     kernel_size=3, stride=1, padding=1)
        self.batch_norm1 = torch.nn.BatchNorm3d(num_features=filters[0], momentum=batch_norm_momentum)

        self.conv2 = torch.nn.Conv3d(in_channels=filters[0], out_channels=filters[1],
                                     kernel_size=3, stride=2, padding=1)
        self.batch_norm2 = torch.nn.BatchNorm3d(num_features=filters[1], momentum=batch_norm_momentum)

        self.conv3 = torch.nn.Conv3d(in_channels=filters[1], out_channels=filters[2],
                                     kernel_size=3, stride=2, padding=1)
        self.batch_norm3 = torch.nn.BatchNorm3d(num_features=filters[2], momentum=batch_norm_momentum)

        self.conv4 = torch.nn.Conv3d(in_channels=filters[2], out_channels=filters[3],
                                     kernel_size=3, stride=2, padding=1)
        self.batch_norm4 = torch.nn.BatchNorm3d(num_features=filters[3], momentum=batch_norm_momentum)

        self.fc = torch.nn.Linear(filters[3] * 216, output_neurons)  # TODO change this input size based on padding
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
