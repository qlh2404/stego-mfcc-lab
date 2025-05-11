import numpy as np
from python_speech_features import mfcc
import librosa
import soundfile as sf  # Them import nay

# Doc dac trung MFCC tu file mfcc_features.npy
mfcc_features = np.load('mfcc_features.npy')

# Doc tin nhan bi mat tu file secret.txt
with open('secret.txt', 'r') as f:
    message = f.read()

# Chuyen tin nhan thanh chuoi bit (moi ky tu thanh 8 bit)
message_bits = ''.join(format(ord(c), '08b') for c in message)

# Sao chep dac trung MFCC de nhung tin
mfcc_stego = mfcc_features.copy()

# Nhung tin vao cac he so MFCC (he so 2-5)
for i, bit in enumerate(message_bits[:len(mfcc_features)]):
    if bit == '0':
        mfcc_stego[i, 2:5] -= 0.1  # Giam nhe gia tri cho bit 0
    else:
        mfcc_stego[i, 2:5] += 0.1  # Tang nhe gia tri cho bit 1

# Luu am thanh da giau tin
audio, sr = librosa.load('sample.wav', sr=16000)
sf.write('stego_audio.wav', audio, sr)  # Su dung soundfile.write thay vi librosa.output

# Luu dac trung MFCC da nhung tin vao file mfcc_stego.npy
np.save('mfcc_stego.npy', mfcc_stego)
print("Da luu am thanh giau tin vao stego_audio.wav")

