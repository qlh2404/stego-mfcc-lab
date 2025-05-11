import librosa
import numpy as np
from python_speech_features import mfcc

# Doc file am thanh mau tu sample.wav
audio, sr = librosa.load('sample.wav', sr=16000)

# Chuan hoa tin hieu am thanh ve khoang [-1, 1]
audio = audio / np.max(np.abs(audio))

# Chia tin hieu thanh cac khung (25ms, chong lan 10ms)
frame_length = int(0.025 * sr)  # Do dai khung: 25ms
hop_length = int(0.010 * sr)   # Do chong lan: 10ms
frames = librosa.util.frame(audio, frame_length=frame_length, hop_length=hop_length).T

# Trich xuat dac trung MFCC tu tin hieu am thanh
mfcc_features = mfcc(audio, sr, winlen=0.025, winstep=0.010, numcep=13)

# Luu dac trung MFCC vao file mfcc_features.npy
np.save('mfcc_features.npy', mfcc_features)
print("Da luu dac trung MFCC vao mfcc_features.npy")

