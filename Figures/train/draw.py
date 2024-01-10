import numpy as np
import matplotlib.pyplot as plt

# 生成模拟的准确率数据
num_epochs = 5000  # 总共迭代的轮数

np.random.seed(23)  # 设置随机种子，以便结果可复现

# 生成准确率数据，剧烈震荡且最后收敛于正值
x = np.linspace(0, 1, num_epochs)
accuracy_values = np.abs(np.sin(101 * np.pi * x) * np.exp(-10 * x))

# 绘制准确率数据
plt.plot(range(num_epochs), accuracy_values)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Model Accuracy Convergence')
plt.ylim(-0.1, 1.1)
plt.grid(True)
plt.show()