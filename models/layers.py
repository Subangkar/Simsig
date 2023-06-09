from torch import nn


class Conv1dSamePad(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.conv = nn.Conv1d(*args, **kwargs)

    def forward(self, input):
        s = self.conv.stride[0]
        input_length = input.shape[2]
        switch = input_length % s
        filter_size = self.conv.kernel_size[0]
        if switch == 0:
            pad_total = max(filter_size - s, 0)
        else:
            pad_total = max(filter_size - (input_length % s), 0)

        left_pad = pad_total // 2
        right_pad = pad_total - left_pad

        input = nn.ConstantPad1d((left_pad, right_pad), 0)(input)
        return self.conv.forward(input)


def conv_pool_block(in_channels, out_channels, kernel_size, pool_size=3, pool_stride=3, pool_pad=0):
    return nn.Sequential(Conv1dSamePad(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size),
                         nn.ReLU(),
                         nn.MaxPool1d(kernel_size=pool_size, padding=pool_pad, stride=pool_stride))


def conv_upsample_block(in_channels, out_channels, kernel_size, padding=None, scale_factor=1):
    conv_layer = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                           padding=padding) if padding else \
        Conv1dSamePad(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)

    return nn.Sequential(conv_layer,
                         nn.ReLU(),
                         nn.Upsample(scale_factor=scale_factor))


def conv_activation_batch_dropout(in_channels, out_channels, kernel_size, strides, activation, padding=None,
                                  dropout_rate=0.00):
    # they used linear in these
    activations = nn.ModuleDict({
        'lrelu': nn.LeakyReLU(),
        'relu': nn.ReLU(),
        'linear': nn.Sequential()
    })

    conv_layer = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=strides,
                           padding=padding) if padding else \
        Conv1dSamePad(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=strides)

    return nn.Sequential(
        conv_layer,
        activations[activation],
        nn.BatchNorm1d(num_features=out_channels),
        nn.Dropout(dropout_rate))
