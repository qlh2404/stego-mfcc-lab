import librosa
import numpy as np
import matplotlib.pyplot as plt

# Doc am thanh goc va am thanh da giu tin
audio_orig, sr = librosa.load('sample.wav', sr=16000)
audio_stego, _ = librosa.load('stego_audio.wav', sr=16000)

# Tinh nho (chenh lech giua am thanh goc va am thanh giu tin)
noise = audio_orig - audio_stego

# Tinh SNR (Signal-to-Noise Ratio)
snr = 10 * np.log10(np.mean(audio_orig**2) / np.mean(noise**2))

# Tinh PSNR (Peak Signal-to-Noise Ratio)
peak = np.max(np.abs(audio_orig))
psnr = 20 * np.log10(peak / np.sqrt(np.mean(noise**2)))

# Luu bao cao chat luong vao file quality_report.txt
with open('quality_report.txt', 'w') as f:
    f.write(f'SNR: {snr:.2f} dB\nPSNR: {psnr:.2f} dB\n')

# Ve bieu do so sanh song am thanh goc va am thanh da giu tin
plt.figure(figsize=(10, 4))
plt.subplot(2, 1, 1)
plt.plot(audio_orig)
plt.title('Am thanh goc')
plt.subplot(2, 1, 2)
plt.plot(audio_stego)
plt.title('Am thanh da giu tin')
plt.tight_layout()

# Luu bieu do vao file audio_comparison.png
plt.savefig('audio_comparison.png')
print("Da luu bao cao chat luong vao quality_report.txt va hinh anh vao audio_comparison.png")

