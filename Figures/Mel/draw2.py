import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

# 读取音频文件
audio_data, sample_rate = librosa.load('bird.wav', sr=None)

# 只保留前两秒的音频数据
duration = 2  # 前两秒的时长
samples_to_keep = int(duration * sample_rate)
audio_data = audio_data[:samples_to_keep]

audio_data = audio_data / np.max(np.abs(audio_data))

spectrogram = np.abs(librosa.stft(audio_data, n_fft=1024))**2

mel_filterbank = librosa.filters.mel(sr=sample_rate, n_fft=1024, n_mels=40)
mel_spectrogram = np.dot(mel_filterbank, spectrogram)

mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

# 绘制频谱图
plt.figure(figsize=(10, 4))

# 原音频的频谱
plt.subplot(1, 2, 1)
librosa.display.specshow(np.log(spectrogram), sr=sample_rate, hop_length=512, x_axis='time', y_axis='linear', cmap='inferno')
plt.title('Spectrogram of Original Audio')
plt.colorbar(format='%+2.0f dB')
plt.xlabel('Time')
plt.ylabel('Frequency')

# 对数压缩后的梅尔频谱
plt.subplot(1, 2, 2)
librosa.display.specshow(mel_spectrogram_db, sr=sample_rate, hop_length=512, x_axis='time', y_axis='mel', cmap='inferno')
plt.title('Mel Spectrogram (Log Compressed)')
plt.colorbar(format='%+2.0f dB')
plt.xlabel('Time')
plt.ylabel('Mel Filterbank Index')

plt.tight_layout()
plt.show()