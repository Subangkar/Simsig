import torch
import torch.nn.functional as F
from torch import nn


def initialize_weights(m):
    if isinstance(m, nn.Conv1d):
        nn.init.kaiming_normal_(m.weight.data, mode='fan_out')
    elif isinstance(m, nn.BatchNorm1d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.bias.data.zero_()


class BottleneckBlock(nn.Module):
    # expansion = 4

    def __init__(self, in_channels, out_channels, stride, cardinality, bottleneck_channels):
        super(BottleneckBlock, self).__init__()
        # self.expansion = expansion

        # bottleneck_channels = cardinality * out_channels // self.expansion

        self.conv1 = nn.Conv1d(
            in_channels,
            bottleneck_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False)
        self.bn1 = nn.BatchNorm1d(bottleneck_channels)

        self.conv2 = nn.Conv1d(
            bottleneck_channels,
            bottleneck_channels,
            kernel_size=9,
            stride=stride,  # downsample with 3x3 conv
            padding=4,
            groups=cardinality,
            bias=False)
        self.bn2 = nn.BatchNorm1d(bottleneck_channels)

        self.conv3 = nn.Conv1d(
            bottleneck_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False)
        self.bn3 = nn.BatchNorm1d(out_channels)

        self.shortcut = nn.Sequential()  # identity
        if in_channels != out_channels:
            self.shortcut.add_module(
                'conv',
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,  # downsample
                    padding=0,
                    bias=False))
            self.shortcut.add_module('bn', nn.BatchNorm1d(out_channels))  # BN

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)), inplace=True)
        y = F.relu(self.bn2(self.conv2(y)), inplace=True)
        y = self.bn3(self.conv3(y))  # not apply ReLU
        y += self.shortcut(x)
        y = F.relu(y, inplace=True)  # apply ReLU after addition
        return y


class ResNext1D(nn.Module):
    def __init__(self):
        super().__init__()

        input_shape = (1, 1, 800)  # configs.INPUT_SHAPE
        n_classes = 2

        base_channels = 64
        self.cardinality = 32

        n_channels = [
            base_channels, base_channels * 2,
                           base_channels * 4,
                           base_channels * 8,
                           base_channels * 16
        ]

        self.conv = nn.Conv1d(
            1,
            n_channels[0],
            kernel_size=49,
            stride=1,
            padding=1,
            bias=False)
        self.bn = nn.BatchNorm1d(n_channels[0])

        self.stage1 = self._make_stage(
            n_channels[0], n_channels[1], 3, stride=4, bottleneck_channels=4 * self.cardinality)
        self.stage2 = self._make_stage(
            n_channels[1], n_channels[2], 4, stride=4, bottleneck_channels=8 * self.cardinality)
        self.stage3 = self._make_stage(
            n_channels[2], n_channels[3], 6, stride=4, bottleneck_channels=12 * self.cardinality)
        self.stage4 = self._make_stage(
            n_channels[3], n_channels[4], 3, stride=4, bottleneck_channels=16 * self.cardinality)

        # compute conv feature size
        with torch.no_grad():
            self.feature_size = self._forward_conv(
                torch.zeros(*input_shape)).view(-1).shape[0]

        self.fc = nn.Linear(self.feature_size, n_classes)

        # initialize weights
        self.apply(initialize_weights)

    def _make_stage(self, in_channels, out_channels, n_blocks, stride, bottleneck_channels):
        stage = nn.Sequential()
        for index in range(n_blocks):
            block_name = 'block{}'.format(index + 1)
            if index == 0:
                stage.add_module(
                    block_name,
                    BottleneckBlock(
                        in_channels,
                        out_channels,
                        stride,  # downsample
                        self.cardinality, bottleneck_channels))
            else:
                stage.add_module(
                    block_name,
                    BottleneckBlock(
                        out_channels,
                        out_channels,
                        1,  # no downsampling
                        self.cardinality, bottleneck_channels))
        return stage

    def _forward_conv(self, x):
        x = F.relu(self.bn(self.conv(x)), inplace=True)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = F.adaptive_avg_pool1d(x, output_size=1)
        return x

    def forward(self, x):
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


if __name__ == "__main__":
    print(torch.__version__)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch = torch.rand(64, 1, 800, dtype=torch.float32, device=device)

    print(batch.shape)
    model = ResNext1D()
    model = model.to(device)
    out1 = model(batch)
    print(out1.to(device='cpu').detach().numpy().shape)
    # print(out2.to(device='cpu').detach().numpy().shape)
