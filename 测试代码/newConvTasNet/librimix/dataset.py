import torch
import torchaudio as ta
import torch.utils.data as Data
import random
import os
import glob
import numpy as np


class LibriMix(Data.Dataset):
    def __init__(self, path_2s1n='', path_1s1n='', ratio=1, seg_len=4, sample_rate=16000):
        assert path_2s1n is not '' or path_1s1n is not ''
        self.path_2s1n = path_2s1n
        self.path_1s1n = path_1s1n
        assert type(ratio) == int
        self.ratio = ratio
        self.speech_type = ['1s1n']
        for _ in range(ratio):
            self.speech_type.append('2s1n')
        self.seg_len = seg_len
        self.sample_rate = sample_rate
        self.utt_len = sample_rate * seg_len
        self.built_speech_list()

    def built_speech_list(self):
        if os.path.exists('speech_list.npy'):
            self.list = np.load('speech_list.npy')
        else:
            list1 = glob.glob(os.path.join(self.path_2s1n, 'mix_both/*.wav'))
            list2 = glob.glob(os.path.join(self.path_1s1n, 'mix_both/*.wav'))
            list = list1 + list2
            list_copy = list.copy()
            for filename in list_copy:
                speech, sr = ta.load(filename)
                if speech.size()[1] / sr < self.seg_len:
                    list.remove(filename)
            self.list = np.random.shuffle(np.array(list))
            np.save('speech_list.npy', self.list)

    def __len__(self):
        return len(self.list)

    def __getitem__(self, idx):
        filename = self.list[idx]
        # get mixture
        mixture, sr = ta.load(filename)
        mixture = ta.transforms.Resample(sr, self.sample_rate)(mixture)
        start = random.randint(0, mixture.size()[1] - self.utt_len)
        stop = start + self.utt_len
        mixture = mixture[:, start:stop]

        # get s1
        s1Filename = filename.replace('mix_both', 's1')
        s1, sr = ta.load(s1Filename)
        s1 = ta.transforms.Resample(sr, self.sample_rate)(s1)
        s1 = s1[:, start:stop]

        # get s2
        s2Filename = filename.replace('mix_both', 's2')
        s2, sr = ta.load(s2Filename)
        s2 = ta.transforms.Resample(sr, self.sample_rate)(s2)
        s2 = s2[:, start:stop]

        return mixture.reshape(1, -1), torch.vstack(s1, s2)


if __name__ == '__main__':
    dataset = LibriMix('../../2s1n', '../../1s1n')
    print(dataset.list)
    print(dataset.speech_type)
