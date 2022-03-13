from model.net import ConvTasNet
# from librimix.librimix_dataset import LibrimixTrainDataset
from librimix.dataset import LibriMix
import torch.utils.data as Data
from loss.pit_wrapper import PITLossWrapper
from loss.sdr import PairwiseNegSDR
import torch
import time
import numpy as np
import os


def run_one_epoch(net, loss_func, optim, dataloader, print_interval):
    train_loss = 0
    net.train()
    for step, (x, y) in enumerate(dataloader):
        torch.cuda.empty_cache()
        x = x.cuda()
        y = y.cuda()

        # 滤除脏数据, 脏数据不参与训练
        if torch.isnan(x).sum() or torch.isinf(x).sum():
            print("warning: nan or inf occurs")
            continue
        out = net(x)
        loss = loss_func(out, y)
        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5)
        optim.step()
        train_loss += loss.data.cpu().numpy()
        # 打印训练日志信息
        if step % print_interval == print_interval - 1:
            mean_loss = train_loss / (step + 1)
            print("step: ", step + 1, "current loss: ", mean_loss)
    return train_loss / (step + 1)


def eval(net, loss_func, dataloader):
    with torch.no_grad():
        net.eval()
        loss_val = 0
        for step, (x, y) in enumerate(dataloader):
            x = x.cuda()
            y = y.cuda()
            out = net(x)
            loss = loss_func(out, y)
            loss_val += loss.data.cpu().numpy()
        loss_val /= step
    return loss_val


def built_solver(parameters, dirPath):
    if not os.path.exists(dirPath):
        os.makedirs(dirPath)
    CHECKPOINT_FILE = os.path.join(dirPath, "ckpt.pth")
    print('-' * 85)
    print("加载模型……")

    # load before
    if os.path.exists(CHECKPOINT_FILE):
        # 恢复训练，加载训练参数
        parameters = np.load(os.path.join(SAVEPATH, "parameters.npy"), allow_pickle=True).item()
        net = ConvTasNet(parameters["N"], parameters["L"], parameters["B"], parameters["Sc"], parameters["H"],
                         parameters["P"],
                         parameters["X"], parameters["R"], parameters["C"], parameters["N_ED"], parameters["mask_act"],
                         parameters["attention"]).cuda()
        print(net)
        optim = torch.optim.Adam(net.parameters(), lr=parameters["LR"])
        checkpoint = torch.load(CHECKPOINT_FILE)
        init_epoch = checkpoint["epoch"]
        optim.load_state_dict(checkpoint["optim"])
        net.load_state_dict(checkpoint['model_state_dict'])
        history = np.load(os.path.join(SAVEPATH, "history.npy"), allow_pickle=True).item()
        if parameters["COSINEANNEALINGLR"]:
            from model.scheduler import CosineAnnealingWarmRestarts
            scheduler = CosineAnnealingWarmRestarts(optim, T_0=5, T_mult=2, last_epoch=init_epoch)
        else:
            scheduler = None
    else:
        save_parameters(parameters, dirPath)
        init_epoch = -1
        history = {
            'train_loss': [],
            'val_loss': [],
            'best_val_loss': 100,
            'pre_val_loss': 100,  # 用于half learning rate
            'no_imprv_count': 0
        }
        net = ConvTasNet(parameters["N"], parameters["L"], parameters["B"], parameters["Sc"], parameters["H"],
                         parameters["P"],
                         parameters["X"], parameters["R"], parameters["C"], parameters["N_ED"], parameters["mask_act"],
                         parameters["attention"]).cuda()
        print(net)
        optim = torch.optim.Adam(net.parameters(), lr=parameters["LR"])
        if parameters["COSINEANNEALINGLR"]:
            from model.scheduler import CosineAnnealingWarmRestarts
            scheduler = CosineAnnealingWarmRestarts(optim, T_0=5, T_mult=2, last_epoch=init_epoch)
        else:
            scheduler = None
    print("学习率：", optim.param_groups[0]['lr'])

    # 优化器
    optimizer = {
        'optim': optim,
        'scheduler': scheduler
    }

    loss_func = PITLossWrapper(PairwiseNegSDR("sisdr"),
                               pit_from='pw_mtx')
    print("加载模型完成……")
    print('-' * 85)
    return net, optimizer, loss_func, history, init_epoch, parameters


def built_dataloader(parameters):
    """
    :return:     :param parameters: train_dataloader,val_dataloader
    """
    print('-' * 85)
    print("加载数据集……")
    train_dataset = LibriMix(parameters["2s1n_train_path"], parameters["1s1n_train_path"],
                             sample_rate=parameters["sample_rate"], seg_len=parameters["seg_len"])
    val_dataset = LibriMix(parameters["2s1n_val_path"], parameters["1s1n_val_path"],
                           sample_rate=parameters["sample_rate"], seg_len=parameters["seg_len"],train=False)

    # train_dataset = LibrimixTrainDataset(speech_path="../LibriSpeech/train-clean-100",
    #                                      noise_path="../wham_noise/tr", data_num=parameters["TRAIN_DATA_NUM"],
    #                                      sample_rate=parameters["SAMPLE_RATE"], sameS=parameters["SAMES"],
    #                                      seg_len=parameters["SEG_LEN"], speech_type=parameters["SPEECH_TYPE"], ratio=parameters["RATIO"])
    # val_dataset = LibrimixTrainDataset(speech_path="../LibriSpeech/dev-clean",
                                    #    noise_path="../wham_noise/cv", train=False,data_num=parameters["VAL_DATA_NUM"],
                                    #      sample_rate=parameters["SAMPLE_RATE"], sameS=parameters["SAMES"],
                                    #      seg_len=parameters["SEG_LEN"], speech_type=parameters["SPEECH_TYPE"], ratio=parameters["RATIO"])

    train_dataloader = Data.DataLoader(train_dataset, batch_size=parameters["BATCH_SIZE"], shuffle=True,
                                       num_workers=parameters["NUM_WORKERS"])
    val_dataloader = Data.DataLoader(val_dataset, batch_size=parameters["BATCH_SIZE"],
                                     num_workers=parameters["NUM_WORKERS"])
    print("加载数据集完成！")
    print('-' * 85)

    return train_dataloader, val_dataloader


def main(parameters, savePath):
    net, optimizer, loss_func, history, init_epoch, parameters = built_solver(parameters, savePath)
    train_dataloader, val_dataloader = built_dataloader(parameters)
    train(train_dataloader, val_dataloader, net, optimizer, loss_func, history, parameters, init_epoch, savePath)


def save_checkpoint(epoch, net, optim, dirPath):
    # 保存最近的模型，以便下一次恢复训练
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': net.state_dict(),
        'optim': optim.state_dict()
    }
    checkpointFilePath = os.path.join(dirPath, 'ckpt.pth')
    torch.save(checkpoint, checkpointFilePath)
    print("checkpoint was saved to ", checkpointFilePath)


def save_parameters(parameters, dirpath):
    paramFilePath = os.path.join(dirpath, "parameters.npy")
    np.save(paramFilePath, parameters)
    print("parameters was save to ", paramFilePath)


def save_history(history, dirpath):
    historyFilePath = os.path.join(dirpath, 'history.npy')
    np.save(historyFilePath, history)
    print("train and val history was saved to ", historyFilePath)


def save_best_weights(net, dirpath):
    bestWeightsFilePath = os.path.join(dirpath, 'best.pth')
    torch.save(net.state_dict(), bestWeightsFilePath)
    print("best weights model was save to ", bestWeightsFilePath)


def train(train_dataloader, val_dataloader, net, optimizer, loss_func, history, parameters, init_epoch, dirPath):
    optim = optimizer['optim']
    scheduler = optimizer['scheduler']
    for epoch in range(init_epoch + 1, parameters['EPOCH']):
        print('-' * 100)
        tic = time.time()
        print("start to train epoch ", epoch)
        train_loss = run_one_epoch(net, loss_func, optim, train_dataloader, parameters['PRINT_LOSS_INTERVAL'])
        print("Epoch: ", epoch + 1, "train loss: ", train_loss)
        val_loss = eval(net, loss_func, val_dataloader)
        print("Epoch: ", epoch + 1, "val loss: ", val_loss)
        if val_loss < history['best_val_loss']:
            best_val_loss = val_loss
            # 保存验证集上最好的模型与最好损失
            save_best_weights(net, dirPath)
            history['best_val_loss'] = best_val_loss
        if parameters["HALF_LR"]:
            if val_loss >= history['pre_val_loss']:
                history['no_imprv_count'] += 1
                if history['no_imprv_count'] == parameters["HALF_LR_EPOCH"]:
                     optim.param_groups[0]['lr'] =  optim.param_groups[0]['lr'] / 2
                    history['no_imprv_count'] = 0
                    print('-' * 85)
                    print("half learning rate! lr = ",  optim.param_groups[0]['lr'])
                    print('-' * 85)
            else:
                history['no_imprv_count'] = 0
                history['pre_val_loss'] = val_loss
        if parameters["COSINEANNEALINGLR"]:
            scheduler.step()
            print('-' * 85)
            print("lr update: ", optim.param_groups[0]['lr'])
            print('-' * 85)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        save_checkpoint(epoch, net, optim, dirPath)
        save_history(history, dirPath)
        print("this epoch costs time ", round((time.time() - tic) / 3600, 4), 'h')
        print('-' * 100)


if __name__ == '__main__':
    # 参数
    parameters = {
        # 模型参数
        "N": 512,
        "L": 40,
        "B": 128,
        "Sc": 128,
        "H": 512,
        "P": 3,
        "X": 8,
        "R": 3,
        "C": 2,
        "N_ED": 2,
        "mask_act": "relu",
        "attention": [1, 1],
        # 训练过程参数
        "EPOCH": 100,
        "LR": 0.001,
        "BATCH_SIZE": 16,
        "PRINT_LOSS_INTERVAL": 50,  # 多少个step打印一次损失
        "NUM_WORKERS": 8,
        "HALF_LR": True,
        "HALF_LR_EPOCH": 5,
        "COSINEANNEALINGLR": False,
        # 语音数据参数
        "sample_rate": 16000,  # 采样率
        "seg_len": 3,  # 每段语音的长度(秒为单位)
        # 固定数据集路径,如果仅采用单种数据集，另一个数据集留空字符串即可
        "2s1n_train_path": "/home/photon/Datasets/Libri2Mix/wav16k/both/train-360/",
        "2s1n_val_path": "/home/photon/Datasets/Libri2Mix/wav16k/both/dev/",
        "1s1n_train_path": "",
        "1s1n_val_path": "",
        # 动态生成数据集参数,使用固定数据集时这里不用管
        "TRAIN_DATA_NUM": 10000,  # 训练数据个数
        "VAL_DATA_NUM": 1200,
        "SAMES": True,
        "RATIO": 2,  # num(2s1n)/num(1s1n)
        "SPEECH_TYPE": "1s1n+2s1n",  # 1s1n,2s1n,1s1n+2s1n
    }
    SAVEPATH = "output"
    main(parameters, SAVEPATH)
