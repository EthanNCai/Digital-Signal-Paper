import numpy as np
import librosa
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei'] # 步骤一（替换sans-serif字体）
plt.rcParams['axes.unicode_minus'] = False   # 步骤二（解决坐标轴负数的负号显示问题）
# 读取音频文件
audio_data, sample_rate = librosa.load('bird.wav', sr=None)


start_time = 1  # 开始时间（以秒为单位）
end_time = 5  # 结束时间（以秒为单位）
start_sample = int(start_time * sample_rate)
end_sample = int(end_time * sample_rate)

audio_data = audio_data[start_sample:end_sample]

# 归一化音频数据
audio_data = audio_data / np.max(np.abs(audio_data))

# 计算原音频的频谱
spectrogram = np.abs(librosa.stft(audio_data, n_fft=1024))**2

# 计算梅尔滤波器组处理后的频谱
mel_filterbank = librosa.filters.mel(sr=sample_rate, n_fft=1024, n_mels=40)
mel_spectrogram = np.dot(mel_filterbank, spectrogram)
mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
# 绘制频谱图


plt.figure(figsize=(7, 2))

# 原音频的频谱
plt.subplot(1, 2, 1)
librosa.display.specshow(np.log(spectrogram), sr=sample_rate, hop_length=512, x_axis='time', y_axis='linear', cmap='viridis')
plt.title('原始音频频语图')
plt.colorbar(format='%+2dB')
plt.xlabel('时间')
plt.ylabel('频率')

# 对数压缩后的梅尔频谱
plt.subplot(1, 2, 2)
librosa.display.specshow(mel_spectrogram_db, sr=sample_rate, hop_length=512, x_axis='time', y_axis='mel', cmap='viridis')
plt.title('梅尔滤波组滤波后的音频频语图(取对数后)')
plt.colorbar(format='%2dB')
plt.xlabel('时间')
plt.ylabel('梅尔滤波器组指数')


plt.show()