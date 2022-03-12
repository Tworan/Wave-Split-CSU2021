# 介绍
语音分离模型Conv-TasNet与LibriMix数据集动态生成的代码实现
- Conv-TasNet论文[《Conv-TasNet: Surpassing Ideal Time-Frequency Magnitude Masking for Speech Separation》](https://arxiv.org/abs/1809.07454)
- LibriMix论文[《LibriMix: An Open-Source Dataset for Generalizable Speech Separation》](https://arxiv.org/abs/2005.11262)

## LibriMix
LibriMix是个很大的数据集，大小超过100G，一些免费的云端计算平台并不能够提供如此大的存储量，若采取动态生成法则可以将存储开销减少到20G以内
### 数据准备
LibiriMix数据集通过结合LibriSpeech与Wham_noise数据集产生，因此需要首先准备这两个数据集
- 下载数据集并将训练集speech(train-clean-100.zip解压后的所有wav文件)和验证集speech(dev-clean.zip解压后的所有wav文件)分别放在两个文件夹下，noise同理
- 在train.py中的built_dataloader函数中的LibrimixTrainDataset中填入speech与noise文件夹路径即可
- 若要使用固定数据集Librimix来训练，在/librimix/dataset中提供了一个类LibriMix，用这个类来代替train.py文件中的LibrimixTrainDataset即可

## Conv-TasNet
### 训练
- 在train.py中的parameters指定模型参数与数据集路径等，基本需要用到的参数都可以进行设置
- 在train.py中还需额外指定保存路径，以便恢复训练和绘制训练历史损失等
- 在draw_loss.ipynb指定训练文件的保存路径，查看训练与验证损失
### 训练快速的ConvTasNet
在train.py中，将
```shell
from model.net import ConvTasNet
```
改为
```shell
from model.fastConvTasNet import ConvTasNet
```
此时模型参数应为

```shell
"N": 512,
"L": 40,
"B": 64,
"Sc": 128,
"H": 512,
"P": 3,
"X": 8,
"R": 4,
"C": 2,
"N_ED": 2,
"mask_act": "relu",
"attention": [1, 1]
```
### 验证
- 在eval.py中指定训练文件的保存路径，然后指定数据集类型
