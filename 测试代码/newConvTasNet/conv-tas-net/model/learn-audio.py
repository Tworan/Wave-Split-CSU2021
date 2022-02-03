import torchaudio
from playsound import playsound

m1, sr_m1 = torchaudio.load("../testData/2/m1.wav")
print(m1.shape)
print(sr_m1)
# playsound("../testData/2/m1.wav")
"""
    重采样
"""
m1 = torchaudio.transforms.Resample(8000, 16000)(m1)
# m1.transforms.Resample(8000,16000)

"""
    数据增强
    混合两段语音，长度不一致则取短
    默认采样率一致
"""


def augMent():
    s1, sr_s1 = torchaudio.load("../testData/1/s1.wav")
    s2, sr_s2 = torchaudio.load("../testData/noise.wav")
    utt_length1 = s1.shape[1]
    utt_length2 = s2.shape[1]
    print(utt_length1, utt_length2)
    s1 = s1.T[:utt_length2].T if utt_length2 < utt_length1 else s1
    s2 = s2.T[:utt_length1].T if utt_length2 > utt_length1 else s2
    print(s1.shape)
    print(s2.shape)
    print(max(s1[0]))
    print(max(s2[0]))
    torchaudio.save("mix.wav",s1+22*s2,8000)

augMent()