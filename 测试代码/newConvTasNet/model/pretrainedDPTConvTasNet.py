import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('/home/photon/Wave-Split-CSU2021/测试代码/newConvTasNet')
from model.norm import GlobLN
import numpy as np
from model.attention import ChannelAttention, SpatialAttention
from asteroid.models import BaseModel
from asteroid.masknn import DPTransformer

pretrained_model = BaseModel.from_pretrained("JorisCos/ConvTasNet_Libri2Mix_sepnoisy_16k")
pretrained_masker = pretrained_model.masker
# pretrained_masker.cuda()

class PMK(nn.Module):
    def __init__(self):
        super(PMK, self).__init__()
        self.top = list(pretrained_masker.children())[0]
        self.bottom = list(pretrained_masker.children())[-1]
        self.mid_layers = nn.ModuleList(list(list(pretrained_masker.children())[1].children())[:16])
        
    def forward(self, mixture):
        mid_out = self.top(mixture)
        batch, _, n_frames = mixture.size()
        skip_connection = torch.tensor([0.0], device=mid_out.device)
        for layer in self.mid_layers:
            mid_out = layer(mid_out)
            residual, skip = mid_out
            skip_connection = skip_connection + skip
            mid_out = residual
        return skip_connection
        

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
        layers.append(nn.Conv1d(1, out_chan, kernel_size=kernel_size, stride=stride, bias=False))
        for _ in range(layer_num - 1):
            layers.append(nn.Conv1d(out_chan, out_chan, kernel_size=1, bias=False))
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
        if layer_num - 1:
            layers.append(nn.Conv1d(in_chan, in_chan//16, kernel_size=1, bias=False))
            in_chan = in_chan//16
        for _ in range(layer_num - 2):
            layers.append(nn.Conv1d(in_chan, in_chan, kernel_size=1, bias=False))
        layers.append(nn.ConvTranspose1d(in_chan, C, kernel_size=kernel_size, stride=stride, bias=False))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # 这里不可以再添加relu或者sigmoid，否则就将结果限制在0~1之间了，而语音信号是可以-1~1之间的
        out = self.net(x)
        return out

class ConvTasNet(nn.Module):
    def __init__(self, N=512, L=40, B=128, Sc=128, H=512, P=3, X=8, R=3, C=2, N_ED=2, mask_act="relu", attention=[True, True], bias=False, freeze=False):
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
        self.tcn = PMK() 
        self.tcn.cuda()
        self.dtn = DPTransformer(
                Sc,
                2,
                n_heads=4,
                ff_hid=512,
                ff_activation='relu',
                chunk_size=100,
                hop_size=None,
                n_repeats=1,
                norm_type='gLN',
                mask_act=mask_act,
                bidirectional=True,
                dropout=0,
            )
        self.decoder = Decoder(Sc, L, L // 2, C, layer_num=N_ED)
        self.conv = nn.Conv1d(N, Sc, kernel_size=(1, ))
        for p in self.tcn.parameters():
            p.require_grads = False

    def forward(self, mixture):
        mixture_e = self.encoder(mixture)  # [B,N,K]
        # mask = self.tcn(mixture_e)  # [B,C,N,K]
        stage_1_output = self.tcn(mixture_e)
        mask = self.dtn(stage_1_output)
        # 这里一定要unsqueeze，否则有逻辑错误 [3,2,128,2999]与[3,128,299]无法相乘，[2,2,128,2999]*[2,128,2999]与[2,2,128,2999]*[2,1,128,2999]不等价
        mixture_e = self.conv(mixture_e)
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
    net = ConvTasNet(attention=[1, 1])
    import time

    input = torch.empty(1, 1, 32000)
    tic = time.time()
    out = net(input)
    print(out.shape)
    print("用时: ", time.time() - tic)
    print(net)
