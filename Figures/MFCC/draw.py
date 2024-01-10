import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display
plt.rcParams['font.sans-serif'] = ['SimHei'] # 步骤一（替换sans-serif字体）
plt.rcParams['axes.unicode_minus'] = False   # 步骤二（解决坐标轴负数的负号显示问题）
start_time = 1 # 开始时间（以秒为单位）
end_time = 3 # 结束时间（以秒为单位）
# Define the audio file paths
audio_files = ['bird.wav', 'bird_01.wav', 'dogbark.wav']
plot_name = ['喜鹊叫声', '麻雀叫声', '狗叫声']


fig, axs = plt.subplots(1, 3, figsize=(7, 3))


for i, audio_file in enumerate(audio_files):

    audio, sr = librosa.load(audio_file, sr=None, offset=start_time, duration=end_time - start_time)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr)
    axs[i].set_title(plot_name[i])
    librosa.display.specshow(mfcc, x_axis='time', ax=axs[i])
    axs[i].figure.colorbar(mappable=axs[i].collections[0], ax=axs[i], format="%+2.0f dB")


plt.tight_layout()

# Display the figure
plt.show()