from model.net import ConvTasNet
from librimix.librimix_dataset import LibrimixTrainDataset
import torch.utils.data as Data
from loss.pit_wrapper import PITLossWrapper
from loss.sdr import PairwiseNegSDR
import torch
import time
import numpy as np
import os

# 数据集超参数
EPOCH = 50
BATCH_SIZE = 12
VAL_INTERVAL = 5000 // BATCH_SIZE  # 每多少个step验证一次
PRINT_LOSS_INTERVAL = 1000 // BATCH_SIZE  # 多少个step打印一次损失
TRAIN_DATA_NUM = 10000  # 训练数据个数
VAL_DATA_NUM = 1200
SAMPLE_RATE = 16000  # 采样率
SEG_LEN = 3  # 每段语音的长度
SAMES = True
NUM_WORKERS = 8
HALF_LR = False
COSINEANNEALINGLR = True
RATIO = 2 # num(2s1n)/num(1s1n)
SPEECH_TYPE = "2s1n" # 1s1n,2s1n,1s1n+2s1n

if HALF_LR and COSINEANNEALINGLR:
    print("half lr and consine annealing lr occurs same time!")
    exit(-1)

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

MAX_NORM = 5
SAVE_EVERY_EPOCH = False
SAVEPATH = "2s1n"
if not os.path.exists(SAVEPATH):
    os.makedirs(SAVEPATH)

print("加载数据集……")
train_dataset = LibrimixTrainDataset(speech_path="../LibriSpeech/train-clean-100",
                                     noise_path="../wham_noise/tr", data_num=TRAIN_DATA_NUM,
                                     sample_rate=SAMPLE_RATE, sameS=SAMES,
                                     seg_len=SEG_LEN,speech_type=SPEECH_TYPE,ratio=RATIO)
val_dataset = LibrimixTrainDataset(speech_path="../LibriSpeech/dev-clean",
                                   noise_path="../wham_noise/cv", data_num=VAL_DATA_NUM,
                                   sample_rate=SAMPLE_RATE, seg_len=SEG_LEN, sameS=SAMES,
                                   train=False,speech_type=SPEECH_TYPE,ratio=RATIO)

train_dataloader = Data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_dataloader = Data.DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
print("加载数据集完成！")

print('-' * 85)
print("加载模型……")
net = ConvTasNet(N, L, B, Sc, H, P, X, R, C, N_ED, mask_act, ATTENTION).cuda()
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
    val_loss_history = list(np.load(os.path.join(SAVEPATH, "val_history.npy")))
    train_loss_history = list(np.load(os.path.join(SAVEPATH, "train_history.npy")))
    time_train = np.load(os.path.join(SAVEPATH,"time.npy"))
    if HALF_LR:
        pre_loss = best_val_loss 
    if COSINEANNEALINGLR:
        from model.scheduler import CosineAnnealingWarmRestarts
        scheduler = CosineAnnealingWarmRestarts(optim, T_0=5,T_mult=2,last_epoch=init_epoch)
else:
    # 加载预训练模型
#     pre_model = torch.load('origin+attention+2en/best.pth')
#     model_dict = net.state_dict()
#     state_dict = {k: v for k, v in pre_model.items() if k in model_dict.keys()}
# #     if N_ED > 1:    # decoder的设计有些特殊，改变结构后预训练权重无法加载
# #         state_dict.pop("decoder.net.0.weight")
#     model_dict.update(state_dict)
#     net.load_state_dict(model_dict)

    init_epoch = -1
    best_val_loss = 100
    val_loss_history = []
    train_loss_history = []
    time_train = 0
    if HALF_LR:
        pre_loss = 100
    if COSINEANNEALINGLR:
        from model.scheduler import CosineAnnealingWarmRestarts
        scheduler = CosineAnnealingWarmRestarts(optim,T_0=5,T_mult=2,last_epoch=init_epoch)
print("学习率：",optim.param_groups[0]['lr'])
print("加载模型完成！")


loss_func = PITLossWrapper(PairwiseNegSDR("sisdr"),
                           pit_from='pw_mtx')

time_start_train = time.time()
print('-' * 85)
print("\n start training!\n")
print('-' * 85)
error_count = 0
no_imprv_count = 0
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
#             errorData = {
#                 "x": x,
#                 "y": y
#             }
#             torch.save(errorData, os.path.join(SAVEPATH, str(error_count) + "_errordata.pth"))
#             error_count += 1
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
        torch.nn.utils.clip_grad_norm_(net.parameters(),MAX_NORM)
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
                # half learning training
                if HALF_LR:
                    if loss_val>=pre_loss:
                        no_imprv_count += 1
                        pre_loss = loss_val
                        if no_imprv_count == 4:
                            optim.state_dict()['param_groups'][0]['lr'] = optim.state_dict()['param_groups'][0]['lr'] / 2
                            no_imprv_count = 0
                    else:
                        no_imprv_count = 0
                        pre_loss = loss_val
    
    # 保存每一个epoch的信息
    # 保存历史损失
    np.save(os.path.join(SAVEPATH, "train_history.npy"), np.array(train_loss_history))
    np.save(os.path.join(SAVEPATH, "val_history.npy"), np.array(val_loss_history))
    # 保存模型
    # 保存最近的模型，以便下一次恢复训练
    if SAVE_EVERY_EPOCH:
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optim': optim.state_dict()
        }
        torch.save(checkpoint, os.path.join(SAVEPATH, str(epoch) + "_ckpt.pth"))
    print('-' * 85)
    print("this Epoch costs  {:.4f}h".format((time.time() - tic) / 3600.0))
    if COSINEANNEALINGLR:
        scheduler.step()
        print("lr update: ",optim.param_groups[0]['lr'])
    print('-' * 85)
    time_train+=(time.time()-tic)
    np.save(os.path.join(SAVEPATH, "time.npy"),time_train)
print("costs total time "(time.time() - time_start_train) / 3600.0, "h")
