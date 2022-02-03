# import numpy as np
# import pyloudnorm as pyln
# import soundfile as sf
#
# speech_list = np.load("train_speech_list.npy")
# meter = pyln.Meter(16000)
# loudness = []
# for item in speech_list:
#     audio, sr = sf.read(item)
#     loudness.append(meter.integrated_loudness(audio))
# loudness = np.array(loudness)
# print("max: ", np.max(loudness))
# print("min: ", np.min(loudness))
# print("average: ",np.average(loudness))

import numpy as np

a = np.arange(3 * 2 * 128 * 2)
a = a.reshape(3, 2, 128, 2)
b = np.arange(3 * 2 * 128)
b = b.reshape(3, 128, 2)
import torch

a = torch.from_numpy(a)
b = torch.from_numpy(b)
# print(a, b)
# x1 = a*b
# print(x1)
print("-" * 87)
b = torch.unsqueeze(b, dim=1)
# print(a, b)
x2 = a*b
print(x2)
print(x1==x2)
