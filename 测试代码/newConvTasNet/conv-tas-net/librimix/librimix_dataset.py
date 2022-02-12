"""
"""

import torch
import torchaudio as ta
import torch.utils.data as Data
import glob
# import soundfile as sf
import os
import numpy as np
import random
from pysndfx import AudioEffectsChain
import pyloudnorm as pyln

# loudness of speech
MAX_LOUDNESS = -25.0
MIN_LOUDNESS = -33.0


class LibrimixTrainDataset(Data.Dataset):
    def __init__(self, speech_path, noise_path, data_num=10000, sample_rate=8000, seg_len=4, train=True, sameS=True,
                 seed=520, speech_type="1s1n+2s1n", ratio=1):
        '''
        
        :param speech_path: speaker speech dir path
        :param noise_path: noise dir path
        :param data_num: number of data each epoch
        :param sample_rate:
        :param seg_len: the length of trained audio
        :param train:
        :param sameS: same speaker
        :param seed: val dataset random seed
        :param speech_type: 1s1n,1s1n+2s1n,2s1n
        :param ratio: num(2s1n)/num(1s1n) should be a positive integer
        '''
        self.speech_path = speech_path
        self.noise_path = noise_path
        self.data_num = data_num
        self.sample_rate = sample_rate
        self.seg_len = seg_len
        self.train = train
        self.sameS = sameS  # 1s1n标签是否为两个相同的声音
        self.seed = seed
        self.speech_type = speech_type
        self.data_type = "train_" if train else "val_"
        self.count = 0  # number of data had been collated this epoch
        self.mix_type = []  # data mixture type
        assert type(ratio) == int
        assert ratio > 0
        for i in range(ratio):
            self.mix_type.append("2s1n")
        self.mix_type.append("1s1n")
        self.speed_type = [0.8, 1.0, 1.2]  # noise speed
        self.utt_len = self.seg_len * self.sample_rate
        self.meter = pyln.Meter(self.sample_rate)  # loudness meter
        self.speech_list, self.noise_list = self.set_speech_list_and_noise_list()
        # 验证集固定, 固定了随机数种子就固定了验证集
        if not train:
            random.seed(self.seed)
            np.random.seed(self.seed)
            indexes = np.random.permutation(self.speech_list.shape[0])
            self.speech_list = self.speech_list[indexes]
            indexes = np.random.permutation(self.noise_list.shape[0])
            self.noise_list = self.noise_list[indexes]

    def set_speech_list_and_noise_list(self):
        '''
        remove audios which time length smaller than seg_len
        :return:
        '''
        if os.path.exists(self.data_type + "speech_list.npy") and os.path.exists(self.data_type + "noise_list.npy"):
            print("built speech list and noise list successfully!")
            speech_list = np.load(self.data_type + "speech_list.npy")
            noise_list = np.load(self.data_type + "noise_list.npy")
            return speech_list, noise_list
        print("start building speech list and noise list ……\nplease wait patiently ……")
        speech_list = glob.glob(self.speech_path + "/*.flac")
        noise_list = glob.glob(self.noise_path + "/*.wav")
        # 不复制的话会有逻辑错误
        speech_list_C = speech_list.copy()
        noise_list_C = noise_list.copy()
        # 保留长度满足要求的语音
        for i, speech in enumerate(speech_list_C):
            if i % 500 == 499:
                print("processing speech no.%d ……" % i)
            audio, sr = ta.load(speech)
            if len(audio[0]) / float(sr) < self.seg_len:
                speech_list.remove(speech)
        for i, noise in enumerate(noise_list_C):
            if i % 500 == 499:
                print("processing noise no.%d ……" % i)
            audio, sr = ta.load(noise)
            if len(audio[0]) / float(sr) < self.seg_len * 1.2:  # the factor 1.2 comes from paper LibriMix
                noise_list.remove(noise)
        speech_list = np.array(speech_list)
        noise_list = np.array(noise_list)
        np.save(self.data_type + "speech_list.npy", speech_list)
        np.save(self.data_type + "noise_list.npy", noise_list)
        print("built speech list and noise list successfully!")
        return speech_list, noise_list

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        mix_type = self.mix_type[random.randint(0, len(self.mix_type)-1)]
        # mixture data has 3types, 1s1n, 2s1n, both
        if self.speech_type != "1s1n+2s1n":
            mix_type = self.speech_type
        if mix_type == "1s1n":
            if self.sameS:  # 1s1n return same speech label
                mixture, source = self.get_1s1n_sameS(idx)
            else:  # return 1 speech and 1 silent speaker
                mixture, source = self.get_1s1n(idx)
        else:
            mixture, source = self.get_2s1n(idx)
        self.count += 1
        # when count to self.data_num, shuffle the list
        if self.count == self.data_num:
            self.count = 0
            # 下一次验证时，验证集保持一致
            if not self.train:
                random.seed(self.seed)
                np.random.seed(self.seed)
            # 训练时则打乱数据，打乱人语音与噪音的对应关系，相当于数据增强
            else:
                indexes = np.random.permutation(self.speech_list.shape[0])
                self.speech_list = self.speech_list[indexes]
                indexes = np.random.permutation(self.noise_list.shape[0])
                self.noise_list = self.noise_list[indexes]

        return mixture.reshape(1, -1), source

    def get_1s1n(self, idx):
        """
        get 1 speech and 1 noise, speech and noise were randomly truncated to self.utt_len
        :param idx:
        :return:
        """

        # get speech
        speech, sr = ta.load(self.speech_list[idx])
        speech = ta.transforms.Resample(sr, self.sample_rate)(speech)
        start = random.randint(0, speech.size()[1] - self.utt_len)
        stop = start + self.utt_len
        speech = speech[:, start:stop].numpy().reshape(-1)
        loudness = self.meter.integrated_loudness(speech)
        dest_loudness = random.uniform(MIN_LOUDNESS, MAX_LOUDNESS)
        # change the loudness according to the paper LibriMix
        speech = pyln.normalize.loudness(speech, loudness, dest_loudness)
        speech = self.check_clip(speech)

        # get silent speaker
        silent_speaker = self.get_silent_speaker(speech)

        # get noise
        speed = self.speed_type[random.randint(0, 2)]
        fx = (AudioEffectsChain().speed(speed))  # change-speed function
        noise, sr = ta.load(self.noise_list[idx])
        noise = noise.reshape(-1).numpy()
        if speed != 1:
            noise = fx(noise)
        noise = torch.from_numpy(noise).reshape(1, -1)
        noise = ta.transforms.Resample(sr, self.sample_rate)(noise.to(torch.float32))
        start = random.randint(0, noise.size()[1] - self.utt_len)
        stop = start + self.utt_len
        noise = noise[:, start:stop].numpy().reshape(-1)
        loudness = self.meter.integrated_loudness(noise)
        dest_loudness = random.uniform(MIN_LOUDNESS - 5, MAX_LOUDNESS - 5)
        # change the loudness according to the paper LibriMix
        noise = pyln.normalize.loudness(noise, loudness, dest_loudness)
        noise = self.check_clip(noise)

        mixture = torch.from_numpy(speech + noise)
        speech = torch.from_numpy(speech)
        silent_speaker = torch.from_numpy(silent_speaker)
        source = torch.vstack((speech, silent_speaker))
        return mixture, source

    def get_2s1n(self, idx):
        """
        get 2 speech and 1 noise, speech and noise were randomly truncated to self.utt_len
        :param idx:
        :return:
        """

        # get speech1
        speech1, sr = ta.load(self.speech_list[idx])
        speech1 = ta.transforms.Resample(sr, self.sample_rate)(speech1)
        start = random.randint(0, speech1.size()[1] - self.utt_len)
        stop = start + self.utt_len
        speech1 = speech1[:, start:stop].numpy().reshape(-1)
        loudness = self.meter.integrated_loudness(speech1)
        dest_loudness = random.uniform(MIN_LOUDNESS, MAX_LOUDNESS)
        # change the loudness according to the paper LibriMix
        speech1 = pyln.normalize.loudness(speech1, loudness, dest_loudness)
        speech1 = self.check_clip(speech1)

        # get speech2
        idx2 = random.randint(self.data_num, len(self.speech_list) - 1)
        basename = os.path.basename(self.speech_list[idx])
        speaker = basename.split("-")[0]
        speaker2 = os.path.basename(self.speech_list[idx2]).split("-")[0]
        # 避免取到同一个speaker的声音
        while speaker2 == speaker:
            idx2 = random.randint(self.data_num, len(self.speech_list) - 1)
            speaker2 = os.path.basename(self.speech_list[idx2]).split("-")[0]

        speech2, sr = ta.load(self.speech_list[idx2])
        speech2 = ta.transforms.Resample(sr, self.sample_rate)(speech2)
        start = random.randint(0, speech2.size()[1] - self.utt_len)
        stop = start + self.utt_len
        speech2 = speech2[:, start:stop].numpy().reshape(-1)
        loudness = self.meter.integrated_loudness(speech2)
        dest_loudness = random.uniform(MIN_LOUDNESS, MAX_LOUDNESS)
        # change the loudness according to the paper LibriMix
        speech2 = pyln.normalize.loudness(speech2, loudness, dest_loudness)
        speech2 = self.check_clip(speech2)

        # get noise
        speed = self.speed_type[random.randint(0, 2)]
        fx = (AudioEffectsChain().speed(speed))  # change-speed function
        noise, sr = ta.load(self.noise_list[idx])
        noise = noise.reshape(-1).numpy()
        if speed != 1:
            noise = fx(noise)
        noise = torch.from_numpy(noise).reshape(1, -1)
        noise = ta.transforms.Resample(sr, self.sample_rate)(noise.to(torch.float32))
        start = random.randint(0, noise.size()[1] - self.utt_len)
        stop = start + self.utt_len
        noise = noise[:, start:stop].numpy().reshape(-1)
        loudness = self.meter.integrated_loudness(noise)
        dest_loudness = random.uniform(MIN_LOUDNESS - 5, MAX_LOUDNESS - 5)
        # change the loudness according to the paper LibriMix
        noise = pyln.normalize.loudness(noise, loudness, dest_loudness)
        noise = self.check_clip(noise)

        mixture = torch.from_numpy(speech1 + speech2 + noise)
        speech1 = torch.from_numpy(speech1)
        speech2 = torch.from_numpy(speech2)
        source = torch.vstack((speech1, speech2))
        return mixture, source

    def get_1s1n_sameS(self, idx):
        """
        get 1 speech and 1 noise, speech and noise were randomly truncated to self.utt_len
        the targets of this mixture include 2 same speech
        :param idx:
        :return:
        """

        # get speech
        speech, sr = ta.load(self.speech_list[idx])
        speech = ta.transforms.Resample(sr, self.sample_rate)(speech)
        start = random.randint(0, speech.size()[1] - self.utt_len)
        stop = start + self.utt_len
        speech = speech[:, start:stop].numpy().reshape(-1)
        loudness = self.meter.integrated_loudness(speech)
        dest_loudness = random.uniform(MIN_LOUDNESS, MAX_LOUDNESS)
        # change the loudness according to the paper LibriMix
        speech = pyln.normalize.loudness(speech, loudness, dest_loudness)
        speech = self.check_clip(speech)

        # get noise
        speed = self.speed_type[random.randint(0, 2)]
        fx = (AudioEffectsChain().speed(speed))  # change-speed function
        noise, sr = ta.load(self.noise_list[idx])
        noise = noise.reshape(-1).numpy()
        if speed != 1:
            noise = fx(noise)
        noise = torch.from_numpy(noise).reshape(1, -1)
        noise = ta.transforms.Resample(sr, self.sample_rate)(noise.to(torch.float32))
        start = random.randint(0, noise.size()[1] - self.utt_len)
        stop = start + self.utt_len
        noise = noise[:, start:stop].numpy().reshape(-1)
        loudness = self.meter.integrated_loudness(noise)
        dest_loudness = random.uniform(MIN_LOUDNESS - 5, MAX_LOUDNESS - 5)
        # change the loudness according to the paper LibriMix
        noise = pyln.normalize.loudness(noise, loudness, dest_loudness)
        noise = self.check_clip(noise)

        mixture = torch.from_numpy(speech + noise)
        speech = torch.from_numpy(speech)
        source = torch.vstack((speech, speech))
        return mixture, source

    def get_silent_speaker(self, speech):
        """
        generate silent speaker.silent speaker is white gaussain noise with energy 70 db below speech
        according to the paper:[Joint Separation and Denoising of Noisy Multi-talker Speech using Recurrent Neural Networks and Permutation Invariant Training]
        (https://arxiv.org/abs/1708.09588)
        :param speech:
        :return:
        """
        below = random.uniform(5, 7)
        d = np.mean(speech ** 2) / (10 ** below)
        silent_speaker = np.random.normal(0, np.sqrt(d), size=speech.shape)
        return silent_speaker

    def check_clip(self, src):
        if np.max(np.abs(src)) >= 1:
            src = src * 0.9 / np.max(np.abs(src))
        return src

    # 用librosa重写速度反而更慢
    # def get_1s1n(self, idx):
    #     """
    #     get 1 speech and 1 noise, speech and noise were randomly truncated to self.utt_len
    #     :param idx:
    #     :return:
    #     """
    #
    #     # get speech
    #     speech, sr = librosa.load(self.speech_list[idx])
    #     speech = librosa.resample(speech, sr, self.sample_rate)
    #     start = random.randint(0, speech.shape[0] - self.utt_len)
    #     stop = start + self.utt_len
    #     speech = speech[start:stop]
    #     loudness = self.meter.integrated_loudness(speech)
    #     dest_loudness = random.uniform(MIN_LOUDNESS, MAX_LOUDNESS)
    #     # change the loudness according to the paper LibriMix
    #     speech = pyln.normalize.loudness(speech, loudness, dest_loudness)
    #
    #     # get noise
    #     speed = self.speed_type[random.randint(0, 2)]
    #     fx = (AudioEffectsChain().speed(speed))  # change-speed function
    #     noise, sr = librosa.load(self.noise_list[idx])
    #     if speed != 1:
    #         noise = fx(noise)
    #     noise = librosa.resample(noise, sr, self.sample_rate)
    #     start = random.randint(0, noise.shape[0] - self.utt_len)
    #     stop = start + self.utt_len
    #     noise = noise[start:stop]
    #     loudness = self.meter.integrated_loudness(noise)
    #     dest_loudness = random.uniform(MIN_LOUDNESS - 5, MAX_LOUDNESS - 5)
    #     # change the loudness according to the paper LibriMix
    #     noise = pyln.normalize.loudness(noise, loudness, dest_loudness)
    #     mixture = torch.from_numpy(speech + noise)
    #     speech = torch.from_numpy(speech)
    #     source = torch.vstack((speech, speech))
    #     return mixture, source
    #
    # def get_2s1n(self, idx):
    #     """
    #     get 2 speech and 1 noise, speech and noise were randomly truncated to self.utt_len
    #     :param idx:
    #     :return:
    #     """
    #
    #     # get speech1
    #     speech1, sr = librosa.load(self.speech_list[idx])
    #     speech1 = librosa.resample(speech1, sr, self.sample_rate)
    #     start = random.randint(0, speech1.shape[0] - self.utt_len)
    #     stop = start + self.utt_len
    #     speech1 = speech1[start:stop]
    #     loudness = self.meter.integrated_loudness(speech1)
    #     dest_loudness = random.uniform(MIN_LOUDNESS, MAX_LOUDNESS)
    #     # change the loudness according to the paper LibriMix
    #     speech1 = pyln.normalize.loudness(speech1, loudness, dest_loudness)
    #
    #     # get speech2
    #     idx2 = random.randint(self.data_num, len(self.speech_list) - 1)
    #     speech2, sr = librosa.load(self.speech_list[idx2])
    #     speech2 = librosa.resample(speech2, sr, self.sample_rate)
    #     start = random.randint(0, speech2.shape[0] - self.utt_len)
    #     stop = start + self.utt_len
    #     speech2 = speech2[start:stop]
    #     loudness = self.meter.integrated_loudness(speech2)
    #     dest_loudness = random.uniform(MIN_LOUDNESS, MAX_LOUDNESS)
    #     # change the loudness according to the paper LibriMix
    #     speech2 = pyln.normalize.loudness(speech2, loudness, dest_loudness)
    #
    #     # get noise
    #     speed = self.speed_type[random.randint(0, 2)]
    #     fx = (AudioEffectsChain().speed(speed))  # change-speed function
    #     noise, sr = librosa.load(self.noise_list[idx])
    #     if speed != 1:
    #         noise = fx(noise)
    #     noise = librosa.resample(noise, sr, self.sample_rate)
    #     start = random.randint(0, noise.shape[0] - self.utt_len)
    #     stop = start + self.utt_len
    #     noise = noise[start:stop]
    #     loudness = self.meter.integrated_loudness(noise)
    #     dest_loudness = random.uniform(MIN_LOUDNESS - 5, MAX_LOUDNESS - 5)
    #     # change the loudness according to the paper LibriMix
    #     noise = pyln.normalize.loudness(noise, loudness, dest_loudness)
    #     mixture = torch.from_numpy(speech1 + speech2 + noise)
    #     speech1 = torch.from_numpy(speech1)
    #     speech2 = torch.from_numpy(speech2)
    #     source = torch.vstack((speech1, speech2))
    #     return mixture, source


if __name__ == '__main__':
    dataset = LibrimixTrainDataset(speech_path="D:/datasets/librispeech/LibriSpeech/train-clean-100",
                                   noise_path="D:/datasets/wham_noise/tr/", data_num=10000)
    dataloader = Data.DataLoader(dataset=dataset, batch_size=2, shuffle=False)
    import time

    tic = time.time()
    for i in range(4000, 5000):
        x, y = dataset[i]
        print(i)
        # pass
    print(time.time() - tic)
