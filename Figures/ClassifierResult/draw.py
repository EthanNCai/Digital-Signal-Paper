import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei'] # 步骤一（替换sans-serif字体）
plt.rcParams['axes.unicode_minus'] = False   # 步骤二（解决坐标轴负数的负号显示问题）
# Bird species and scores
bird_species = ['喜鹊', '麻雀', '知更鸟', '哈士奇狗叫', '猪叫声']
scores = [0.928, 0.632, 0.601, 0.052, 0.013]
plt.figure(figsize=(4, 4))
# Plotting the bar chart
plt.bar(bird_species, scores,  color='navy')

# Adding labels and title
plt.xlabel('叫声类别')
plt.ylabel('分类得分')
plt.title('一份喜鹊叫声样本的分类结果')

# Displaying the chart
plt.show()