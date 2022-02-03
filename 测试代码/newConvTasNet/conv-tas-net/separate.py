from model.net import ConvTasNet
from librimix.librimix_dataset import LibrimixTrainDataset
import torch
import soundfile as sf
import time
from loss.pit_wrapper import PITLossWrapper
from loss.sdr import PairwiseNegSDR
import numpy as np



net = ConvTasNet(L=32)
net.load_state_dict(torch.load("output/best.pth"))

input,sr = sf.read("mixture.wav")
y1, sr = sf.read("y1.wav")
y2, sr = sf.read("y2.wav")
y = np.vstack((y1,y2))
# 一次只能喂一个数据     input:[T] ,output: [2,T]    ndarray
out = net.separate(input)
sf.write("out1.wav", out[0], 16000)
sf.write("out2.wav", out[1], 16000)


loss_func = PITLossWrapper(PairwiseNegSDR("sisdr"),
                           pit_from='pw_mtx')
out = torch.from_numpy(out.reshape(1,2,-1))
y = torch.from_numpy(y.reshape(1,2,-1))

# input [bs,1,T] output [bs,2,T]
# when mixture is 1s1n, y provide 2 same speech
loss = loss_func(out,y)
print(loss)


