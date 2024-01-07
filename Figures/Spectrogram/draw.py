import librosa
import matplotlib.pyplot as plt
import matplotlib.cm as cm

plt.rcParams['font.sans-serif'] = ['SimHei'] # 步骤一（替换sans-serif字体）
plt.rcParams['axes.unicode_minus'] = False   # 步骤二（解决坐标轴负数的负号显示问题）
def plot_spectrogram(audio_path, subplot_num, subtitle):
    # 读取音频文件
    waveform, sample_rate = librosa.load(audio_path, sr=None)

    # 提取前一秒的音频
    start_time = 1  # 开始时间（以秒为单位）
    end_time = 5  # 结束时间（以秒为单位）
    start_sample = int(start_time * sample_rate)
    end_sample = int(end_time * sample_rate)
    waveform_slice = waveform[start_sample:end_sample]

    # 计算音频的短时傅里叶变换（STFT）
    n_fft = 2048 * 2  # FFT窗口大小
    hop_length = 512  # 帧之间的跳跃长度
    stft = librosa.stft(waveform_slice, n_fft=n_fft, hop_length=hop_length)

    # 将STFT转换为分贝刻度
    spectrogram = librosa.amplitude_to_db(abs(stft))

    # 绘制语谱图
    plt.subplot(3, 1, subplot_num)
    cmap = cm.get_cmap('viridis')  # 获取黑紫黄色映射
    librosa.display.specshow(
        spectrogram, sr=sample_rate, hop_length=hop_length, cmap=cmap, x_axis='time', y_axis='log'
    )
    plt.colorbar(format='%+2.0f dB', label='强度 (dB)')
    plt.title(subtitle)
    plt.xlabel('')
    plt.yticks([])
    plt.xticks(minor=False)
    plt.ylabel('频率')


# 画一个包含三个子图的大图
plt.figure(figsize=(4, 4))

# 画第一个子图
plt.subplot(3, 1, 1)
audio_path_subway = 'bird.wav'
plot_spectrogram(audio_path_subway, 1, '喜鹊叫声', )

# 画第二个子图
plt.subplot(3, 1, 2)
audio_path_bus1 = 'bird_01.wav'
plot_spectrogram(audio_path_bus1, 2, '麻雀叫声')

plt.subplot(3, 1, 3)
audio_path_bus1 = 'dogbark.wav'
plot_spectrogram(audio_path_bus1, 3, '狗叫声')

plt.tight_layout()
plt.show()
