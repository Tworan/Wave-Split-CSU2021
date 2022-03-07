import torch
import torch.nn as nn

"""
    according to the paper  [CBAM: Convolutional Block Attention Module]
                            (https://arxiv.org/pdf/1807.06521.pdf)
"""


class ChannelAttention(nn.Module):
    def __init__(self, in_chan, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(nn.Conv1d(in_chan, in_chan // ratio, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv1d(in_chan // ratio, in_chan, 1, bias=False))

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        out = torch.sigmoid(out)
        return out * x


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv1d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x):
        """
        :param x: [bs,in_chan,T]
        :return: out: [bs,1,T]
        """
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        out = torch.sigmoid(out)
        return out * x


if __name__ == '__main__':
    input = torch.rand(1, 128, 3999)
    net = ChannelAttention(128)
    out = net(input)  # [bs,in_chan,1]
    print(out.shape)

    net2 = SpatialAttention(7)
    out2 = net2(input)
    print(out2.shape)
