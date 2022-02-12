import torch
from librimix.librimix_dataset import LibrimixTrainDataset
import torch.utils.data as Data
from model.net import ConvTasNet
from loss.pit_wrapper import PITLossWrapper
from loss.sdr import PairwiseNegSDR
import time

# 数据集超参数
SAMPLE_RATE = 16000
SEG_LEN = 3
DATA_NUM = 1200
SAMES = True
BATCH_SIZE = 16
NUM_WORKERS = 8

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
ATTENTION = [1, 1]
CHECKPOINTPATH = "origin+attention+2en+r2/best.pth"

# 加载模型
print("加载模型……")
net = ConvTasNet(N, L, B, Sc, H, P, X, R, C, N_ED, mask_act, attention=ATTENTION).cuda()
net.load_state_dict(torch.load(CHECKPOINTPATH))
print("加载模型完成！")

print("加载数据集……")
dataset = LibrimixTrainDataset(speech_path="D:/datasets/librispeech/LibriSpeech/dev-clean",
                               noise_path="D:/datasets/wham_noise/cv", data_num=DATA_NUM,
                               sample_rate=SAMPLE_RATE, seg_len=SEG_LEN, sameS=SAMES,
                               train=False, val_type="1s1n")
dataloader = Data.DataLoader(dataset, batch_size=BATCH_SIZE,num_workers=NUM_WORKERS)
print("加载数据集完成！")
loss_func = PITLossWrapper(PairwiseNegSDR("sisdr"),
                           pit_from='pw_mtx')

print("*" * 90)
print("start eval……")
print("*" * 90)
tic = time.time()
loss_val = 0
with torch.no_grad():
    net.eval()
    for i, (x, y) in enumerate(dataloader):
        if i % 50 == 49:
            print("processing No.", i)
        x, y = x.cuda(), y.cuda()
        pred = net(x)
        loss = loss_func(pred, y)
        loss_val += loss.data.cpu().numpy()
    print("loss = ", loss_val / (i + 1))
    print("用时：", time.time() - tic, "s")

# tic = time.time()
# loss_val = 0
# with torch.no_grad():
#     net.eval()
#     for i, (x, y) in enumerate(dataloader):
#         if i % 50 == 49:
#             print("processing No.", i)
#         x, y = x.cuda(), y.cuda()
#         pred = net(x)
#         loss = loss_func(pred, y)
#         loss_val += loss.data.cpu().numpy()
#     print("loss = ", loss_val / (i + 1))
#     print("用时：", time.time() - tic, "s")
