from model.net import ConvTasNet
from librimix.librimix_dataset import LibrimixTrainDataset
import torch.utils.data as Data
from loss.pit_wrapper import PITLossWrapper
from loss.sdr import PairwiseNegSDR
import torch
import time
import numpy as np
import os

# 一些超参数
EPOCH = 100
VAL_INTERVAL = 1250  # 每多少个step验证一次
PRINT_LOSS_INTERVAL = 250  # 多少个step打印一次损失
TRAIN_DATA_NUM = 10000  # 训练数据个数
VAL_DATA_NUM = 300
SAMPLE_RATE = 16000  # 采样率
SEG_LEN = 3  # 每段语音的长度
BATCH_SIZE = 2
FP16_TRAIN = False
MAX_NORM = 5  # Gradient norm threshold to clip
sames = False
SAVEPATH = "output_diffs"

if not os.path.exists(SAVEPATH):
    os.makedirs(SAVEPATH)
if FP16_TRAIN:
    from torch.cuda.amp import autocast, GradScaler

    scaler = GradScaler()

stride = 32 if SAMPLE_RATE == 16000 else 16

train_dataset = LibrimixTrainDataset(speech_path="D:/datasets/librispeech/LibriSpeech/train-clean-100",
                                     noise_path="D:/datasets/wham_noise/tr", data_num=TRAIN_DATA_NUM,
                                     sample_rate=SAMPLE_RATE, sameS=sames,
                                     seg_len=SEG_LEN)
val_dataset = LibrimixTrainDataset(speech_path="D:/datasets/librispeech/LibriSpeech/dev-clean",
                                   noise_path="D:/datasets/wham_noise/cv", data_num=VAL_DATA_NUM,
                                   sample_rate=SAMPLE_RATE, seg_len=SEG_LEN, sameS=sames,
                                   train=False)

train_dataloader = Data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = Data.DataLoader(val_dataset, batch_size=BATCH_SIZE)

net = ConvTasNet(L=stride).cuda()
print(net)
CHECKPOINT_FILE = os.path.join(SAVEPATH, "ckpt.pth")
optim = torch.optim.Adam(net.parameters(), lr=0.001)

# load before
if os.path.exists(CHECKPOINT_FILE):
    checkpoint = torch.load(CHECKPOINT_FILE)
    init_epoch = checkpoint["epoch"]
    optim.load_state_dict(checkpoint["optim"])
    net.load_state_dict(checkpoint['model_state_dict'])
    best_val_loss = np.load(os.path.join(SAVEPATH, "best_val_loss.npy"))
    val_loss_history = list(np.load(os.path.join(SAVEPATH, "train_history.npy")))
    train_loss_history = list(np.load(os.path.join(SAVEPATH, "val_history.npy")))
else:
    # 加载预训练模型
    net.load_state_dict(torch.load(os.path.join(SAVEPATH, "best.pth")))
    init_epoch = -1
    best_val_loss = 100
    val_loss_history = []
    train_loss_history = []

loss_func = PITLossWrapper(PairwiseNegSDR("sisdr"),
                           pit_from='pw_mtx')

time_start_train = time.time()
print('-' * 85)
print("\n start training!\n")
print('-' * 85)
error_count = 0
for epoch in range(init_epoch + 1, EPOCH):
    tic = time.time()
    train_loss = 0
    for step, (x, y) in enumerate(train_dataloader):
        torch.cuda.empty_cache()
        net.train()
        x = x.cuda()
        y = y.cuda()

        # 滤除脏数据, 脏数据不参与训练
        if torch.isnan(x).sum() or torch.isinf(x).sum():
            print("warning: nan or inf occurs")
            errorData = {
                "x": x,
                "y": y
            }
            torch.save(errorData, os.path.join(SAVEPATH, str(error_count) + "_errordata.pth"))
            error_count += 1
            continue
        # 混合精度训练
        # if FP16_TRAIN:
        #     with autocast():
        #         out = net(x)
        #         loss = loss_func(out, y)
        #     scaler.scale(loss).backward()
        #     scaler.step(optim)
        #     scaler.update()
        out = net(x)
        loss = loss_func(out, y)
        optim.zero_grad()
        loss.backward()
        optim.step()
        train_loss += loss.data.cpu().numpy()
        if step % PRINT_LOSS_INTERVAL == PRINT_LOSS_INTERVAL - 1:
            mean_loss = train_loss / ((step % PRINT_LOSS_INTERVAL) + 1)
            print("Epoch: ", epoch + 1, "step: ", step + 1, "current loss: ", mean_loss)
            train_loss = 0
            train_loss_history.append(mean_loss)

        # 验证
        if step % VAL_INTERVAL == VAL_INTERVAL - 1:
            with torch.no_grad():
                net.eval()
                loss_val = 0
                for _, (x, y) in enumerate(val_dataloader):
                    x = x.cuda()
                    y = y.cuda()
                    out = net(x)
                    loss = loss_func(out, y)
                    loss_val += loss.data.cpu().numpy()
                loss_val /= float(VAL_DATA_NUM / BATCH_SIZE)
                if loss_val < best_val_loss:
                    best_val_loss = loss_val
                    # 保存验证集上最好的模型与最好损失
                    torch.save(net.state_dict(), os.path.join(SAVEPATH, "best.pth"))
                    np.save(os.path.join(SAVEPATH, "best_val_loss.npy"), best_val_loss)

                print("Epoch: ", epoch + 1, "step: ", step + 1, "val loss: ", loss_val)
                val_loss_history.append(loss_val)
                # 保存最近的模型，以便下一次恢复训练
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': net.state_dict(),
                    'optim': optim.state_dict()
                }
                torch.save(checkpoint, CHECKPOINT_FILE)

    # 保存每一个epoch的信息
    # 保存历史损失
    np.save(os.path.join(SAVEPATH, "train_history.npy"), np.array(train_loss_history))
    np.save(os.path.join(SAVEPATH, "val_history.npy"), np.array(val_loss_history))
    # 保存模型
    # 保存最近的模型，以便下一次恢复训练
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': net.state_dict(),
        'optim': optim.state_dict()
    }
    torch.save(checkpoint, os.path.join(SAVEPATH, str(epoch) + "_ckpt.pth"))
    print('-' * 85)
    print("this Epoch costs  {:.4f}h".format((time.time() - tic) / 3600.0))
    print('-' * 85)

print("costs total time "(time.time() - time_start_train) / 3600.0, "h")
