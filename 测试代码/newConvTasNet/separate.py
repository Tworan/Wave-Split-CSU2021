from model.net import ConvTasNet
from librimix.librimix_dataset import LibrimixTrainDataset
import torch
import soundfile as sf
import time
from loss.pit_wrapper import PITLossWrapper
from loss.sdr import PairwiseNegSDR
import numpy as np

SAMPLE_RATE = 16000
# 模型超参数
N = 128
L = 32 if SAMPLE_RATE == 16000 else 16
B = 128
Sc = 128
H = 256
P = 3
X = 8
R = 3
C = 2
N_ED = 1
mask_act = "relu"
CHECKPOINTPATH = "output_attention/best.pth"
ATTENTION = [True, True]

net = ConvTasNet(N, L, B, Sc, H, P, X, R, C, N_ED, mask_act, ATTENTION)
net.load_state_dict(torch.load(CHECKPOINTPATH))
net.eval()
val_dataset = LibrimixTrainDataset(speech_path="D:/datasets/librispeech/LibriSpeech/dev-clean",
                                   noise_path="D:/datasets/wham_noise/cv", data_num=200,
                                   sample_rate=16000, seg_len=4,
                                   train=False)
loss_func = PITLossWrapper(PairwiseNegSDR("sisdr"),
                           pit_from='pw_mtx')

# 测试sdr
# x, y = val_dataset[0]
# x = x.reshape(1, 1, -1)
# y = y.reshape(1, 2, -1)
# out = net(x)
# print(loss_func(out, y))

''' 测试 '''

x, y = val_dataset[160]
x = x.numpy().reshape(-1)
y = y.numpy().reshape(2, -1)
tic = time.time()
out = net.separate(x)
print(time.time() - tic)
sf.write("out1.wav", out[0], 16000)
sf.write("out2.wav", out[1], 16000)
sf.write("mixture.wav", x, 16000)
sf.write("y1.wav", y[0], 16000)
sf.write("y2.wav", y[1], 16000)
