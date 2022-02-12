from model.net import ConvTasNet
# from librimix.librimix_dataset import LibrimixTrainDataset
import torch
import soundfile as sf
import time
# from loss.pit_wrapper import PITLossWrapper
# from loss.sdr import PairwiseNegSDR
import numpy as np

SAMPLE_RATE = 16000
# 模型超参数
N = 512
L = 32 if SAMPLE_RATE == 16000 else 16
B = 128
Sc = 128
H = 512
P = 3
X = 8
R = 3
C = 2
N_ED = 2
mask_act = "relu"
CHECKPOINTPATH = "origin+attention+2en+r2/best.pth"
ATTENTION = [True, True]

net = ConvTasNet(N, L, B, Sc, H, P, X, R, C, N_ED, mask_act, ATTENTION)
net.load_state_dict(torch.load(CHECKPOINTPATH))
net.eval()

''' 测试 '''
x,sr = sf.read("mixture.wav")
tic = time.time()
# 采样率需要使16k,x需要是numpy数组，一次只能一条语音
out = net.separate(x)
print(time.time() - tic)
sf.write("out1.wav", out[0], 16000)
sf.write("out2.wav", out[1], 16000)
