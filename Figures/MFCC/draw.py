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

# Create a figure with three subplots
fig, axs = plt.subplots(1, 3, figsize=(7, 3))

# Iterate over the audio files
for i, audio_file in enumerate(audio_files):
    # Load the audio file
    audio, sr = librosa.load(audio_file, sr=None, offset=start_time, duration=end_time - start_time)

    # Extract MFCC features
    mfcc = librosa.feature.mfcc(y=audio, sr=sr)

    # Plot the MFCC features in the corresponding subplot
    axs[i].set_title(plot_name[i])
    librosa.display.specshow(mfcc, x_axis='time', ax=axs[i])

    # Add a colorbar to each subplot
    axs[i].figure.colorbar(mappable=axs[i].collections[0], ax=axs[i], format="%+2.0f dB")

# Set the overall title for the figure


# Adjust the spacing between subplots
plt.tight_layout()

# Display the figure
plt.show()