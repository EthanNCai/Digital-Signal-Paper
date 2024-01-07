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

# 绘制频谱图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 2))

# 左边子图：原音频的频谱
ax1.imshow(np.log(spectrogram), aspect='auto', origin='lower', cmap='viridis')
ax1.set_title('原始频语图')
ax1.set_xlabel('')
ax1.set_ylabel('原始频率')

# 右边子图：梅尔滤波器组处理后的频谱
ax2.imshow(np.log(mel_spectrogram), aspect='auto', origin='lower', cmap='viridis')
ax2.set_title('梅尔滤波后的频语图')
ax2.set_xlabel('')
ax2.set_ylabel('梅尔滤波器组指数')


plt.show()