import torch
from librimix.librimix_dataset import LibrimixTrainDataset
import torch.utils.data as Data
from model.net import ConvTasNet
from loss.pit_wrapper import PITLossWrapper
from loss.sdr import PairwiseNegSDR
import time
import os

path = "output/no_skip"
checkpointPath = ps.path.join(path,'best.pth')
parameters = np.load(os.path.join(path,"parameters.npy"),allow_pickle=True).item()

# 加载模型
print("加载模型……")
net = ConvTasNet(parameters["N"],parameters["L"],parameters["B"],parameters["Sc"],parameters["H"],parameters["P"],
        parameters["X"], parameters["R"],parameters["C"], parameters["N_ED"], parameters["mask_act"], parameters["attention"]).cuda()
net.load_state_dict(torch.load(checkpointPath))
print("加载模型完成！")

print("加载数据集……")
dataset = LibrimixTrainDataset(speech_path="D:/datasets/librispeech/LibriSpeech/dev-clean",
                               noise_path="D:/datasets/wham_noise/cv", data_num=1000,
                               sample_rate=16000, seg_len=3, sameS=True,
                               train=False, val_type="1s1n")
dataloader = Data.DataLoader(dataset, batch_size=64,num_workers=8)
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
