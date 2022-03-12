import torch
import torch.nn as nn
import torch.nn.functional as F
from model.norm import GlobLN
import numpy as np
from model.attention import ChannelAttention, SpatialAttention


class Conv1DBlock(nn.Module):
    """ 1-D Conv block in Conv-TasNet """

    def __init__(self, in_chan, hid_chan, skip_chan, kernel_size, padding, dilation, attention=False):
        super(Conv1DBlock, self).__init__()
        in_conv = nn.Conv1d(in_chan, hid_chan, 1)
        relu = nn.PReLU()
        norm = GlobLN(hid_chan)
        D_Conv = nn.Conv1d(hid_chan, hid_chan, kernel_size=kernel_size, padding=padding, dilation=dilation,
                           groups=hid_chan)
        self.res_conv = nn.Conv1d(hid_chan, in_chan, 1)
        self.skip_chan = skip_chan
        if self.skip_chan:
            self.skip_conv = nn.Conv1d(hid_chan, skip_chan, 1)
        self.shared_block = nn.Sequential(
            in_conv,
            relu,
            norm,
            D_Conv,
            relu,
            norm
        )
        self.attention = attention
        if self.attention:
            self.res_ca = ChannelAttention(in_chan)
            self.res_sa = SpatialAttention()
            if self.skip_chan:
                self.skip_ca = ChannelAttention(skip_chan)
                self.skip_sa = SpatialAttention()

    def forward(self, x):
        share_out = self.shared_block(x)
        res_out = self.res_conv(share_out)
        if self.attention:
            res_out = self.res_ca(res_out)
            res_out = self.res_sa(res_out)
        res_out = x + res_out
        if self.skip_chan:
            skip_out = self.skip_conv(share_out)
            if self.attention:
                skip_out = self.skip_ca(skip_out)
                skip_out = self.skip_sa(skip_out)
            return res_out, skip_out
        return res_out

class Encoder(nn.Module):
    def __init__(self, out_chan, kernel_size, stride, layer_num=1, attention=False):
        """
        1-D Conv Encoder
        :param out_chan: number of filters in autoencoder
        :param kernel size:
        :param stride: length of filters(in samples)
        :param layer_num: number of layers of encoder
        """
        super(Encoder, self).__init__()
        layers = []
        layers.append(nn.Conv1d(1,out_chan//32,4,2,1))
        layers.append(nn.Conv1d(out_chan//32, out_chan, kernel_size=kernel_size, stride=stride, bias=False))
        self.net = nn.Sequential(*layers)
        self.attention = attention
        if self.attention:
            self.ca = ChannelAttention(out_chan)
            self.sa = SpatialAttention()

    def forward(self, x):
        """
        :param x: [BS,1,T], BS is batch size, T is #samples
        :return: ouput: [BS,N,K], K = (T-L)/(L/2)+1 = 2T/L-1 when P = 3
        """
        out = F.relu(self.net(x))  # [BS,1,T] --> [BS,N,K]
        if self.attention:
            out = self.ca(out)
            out = self.sa(out)
        return out


class Decoder(nn.Module):
    def __init__(self, in_chan, kernel_size, stride, C, layer_num=1):
        '''
        decoder, 还原mask与原始输入数据一致
        :param in_chan:
        :param kernel_size:
        :param stride:
        :param C: num of speakers
        '''
        super(Decoder, self).__init__()
        layers = []
        in_chan = C * in_chan
#         if layer_num - 1:
#             layers.append(nn.Conv1d(in_chan, in_chan//16, kernel_size=1, bias=False))
#             in_chan = in_chan//16
#         for _ in range(layer_num - 2):
#             layers.append(nn.Conv1d(in_chan, in_chan, kernel_size=1, bias=False))
        layers.append(nn.ConvTranspose1d(in_chan, in_chan//32, kernel_size=kernel_size, stride=stride, bias=False))
        layers.append(nn.ConvTranspose1d(in_chan//32,C,4,2,1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # 这里不可以再添加relu或者sigmoid，否则就将结果限制在0~1之间了，而语音信号是可以-1~1之间的
        out = self.net(x)
        return out


class TCN(nn.Module):
    def __init__(self, in_chan, hid_chan, bn_chan, skip_chan, n_repeat, n_conv1d, kernel_size, C, mask_act,
                 attention=False):
        """
        Temporal Convolutional network used in Conv-TasNet.
        :param in_chan: number of channels in encoder
        :param hid_chan: number of channels in D-Conv
        :param bn_chan: number of channels in bottleneck
        :param skip_chan: skip connection channel
        :param n_repeat: repeat times
        :param n_conv1d: number of 1-D conv block in every repeat
        :param kernel_size:
        :param C: number of speakers
        :param mask_act:
        """
        super(TCN, self).__init__()
        self.C = C
        self.skip_chan = skip_chan
        self.in_chan = in_chan
        layer_norm = GlobLN(in_chan)
        bottleneck_conv = nn.Conv1d(in_chan, bn_chan, kernel_size=1)
        self.bottleneck = nn.Sequential(layer_norm, bottleneck_conv)
        self.tcn = nn.ModuleList()
        for r in range(n_repeat):
            for x in range(n_conv1d):
                dilation = 2 ** x
                padding = (kernel_size - 1) * dilation // 2
                self.tcn.append(Conv1DBlock(bn_chan,
                                            hid_chan,
                                            skip_chan,
                                            kernel_size,
                                            padding,
                                            dilation,
                                            attention))
        if self.skip_chan:
            self.mask_conv = nn.Sequential(nn.PReLU(),
                                       nn.Conv1d(skip_chan, C * in_chan, 1, bias=False)
                                       )
        else:
            self.mask_conv = nn.Sequential(nn.PReLU(),
                                       nn.Conv1d(bn_chan, C * in_chan, 1, bias=False)
                                       )
        self.mask_act = get_activation(mask_act)

    def forward(self, mixture_e):
        '''
        :param mixture_e: mixture after encoder
        :return:
        '''
        batch, _, n_frames = mixture_e.size()
        output = self.bottleneck(mixture_e)  # mixture after bottleneck
        skip_connection = torch.tensor([0.0], device=output.device)
        for layer in self.tcn:
            # Common to w. skip and w.o skip architectures
            tcn_out = layer(output)
            if self.skip_chan:
                residual, skip = tcn_out
                skip_connection = skip_connection + skip
            else:
                residual = tcn_out
            output = residual
        # Use residual output when no skip connection
        mask_in = skip_connection if self.skip_chan else output
        mask = self.mask_conv(mask_in)  # [B, C * in_chan, frames]
        mask = self.mask_act(mask)  # 放缩到0,1之间
        mask = mask.view(batch, self.C, self.in_chan, n_frames)
        return mask


class ConvTasNet(nn.Module):
    def __init__(self, N=512, L=40, B=64, Sc=128, H=512, P=3, X=8, R=4, C=2, N_ED=1, mask_act="relu", attention=[False,False]):
        """
        :param N: number of filters in autoencoder
        :param L: length of filters(in samples)
        :param B: number of channels in bottleneck and the residual paths' 1 X 1-conv blocks
        :param Sc: number of channels in skip-connection paths' 1 X 1-conv blocks
        :param H: number of channels in convolutional blocks
        :param P: kernel size in convolutional blocks
        :param X: number of convolutional in each repeat
        :param R: number of repeats
        :param C: number of speakers
        :param N_ED: layers number of encoder and decoder
        :param mask_act: mask activation
        """
        super(ConvTasNet, self).__init__()
        self.N = N
        self.L = L
        self.B = B
        self.Sc = Sc
        self.H = H
        self.P = P
        self.X = X
        self.R = R
        self.mask_act = mask_act
        self.encoder = Encoder(N, L, L // 2, layer_num=N_ED, attention=attention[0])
        self.tcn = TCN(N, H, B, Sc, R, X, P, C, mask_act, attention=attention[1])
        self.decoder = Decoder(N, L, L // 2, C, layer_num=N_ED)

    def forward(self, mixture):
        mixture_e = self.encoder(mixture)  # [B,N,K]
        mask = self.tcn(mixture_e)  # [B,C,N,K]
        # 这里一定要unsqueeze，否则有逻辑错误 [3,2,128,2999]与[3,128,299]无法相乘，[2,2,128,2999]*[2,128,2999]与[2,2,128,2999]*[2,1,128,2999]不等价
        mixture_e = torch.unsqueeze(mixture_e, dim=1)  # [B,1,N,K]
        B, C, N, K = mask.size()
        out = mask * mixture_e  # [B,C,N,K]
        out = out.view(B, C * N, K)  # [B,C*N,K]
        out = self.decoder(out)
        return out

    def separate(self, x):
        """
        :param x: numpy array [T], T is the length of the samples of this image
        :return:
        """
        assert len(x.shape) == 1
        # encoder这一步决定了T必须pad到能被L/2整除，否则输出长度与标签长度便可能发生冲突
        origin_length = x.shape[0]
        pad = self.L // 2 - origin_length % (self.L // 2)
        if pad != self.L // 2:
            dest_length = origin_length + pad
            input = np.zeros((1, dest_length), dtype=np.float32)
            input[0, :origin_length] = x
        else:
            input = x
        x = torch.from_numpy(x)
        input = torch.from_numpy(input.reshape(1, 1, -1))
        output = self.forward(input)
        # 利用这一步改善语音质量,非常巧妙的一步
        output = output[:, :, :origin_length]
        output *= x.abs().sum() / (output.abs().sum())
        return output[0].detach().numpy()


def get_activation(activation):
    if activation == "relu":
        return nn.ReLU()
    if activation == "sigmoid":
        return nn.Sigmoid()
    if activation == "leakyrelu":
        return nn.LeakyReLU()
    if activation == "tanh":
        return nn.Tanh()
    if activation == "prelu":
        return nn.PReLU()


if __name__ == '__main__':
    net = ConvTasNet(attention=True)
    import time

    input = torch.empty(1, 1, 32000)
    tic = time.time()
    out = net(input)
    print(out.shape)
    print("用时: ", time.time() - tic)
    print(net)
