
import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

# pip install scikit-learn


# 假设你的张量是这个
tensor = torch.randn(8, 20, 512)  # 示例数据

# 将张量重新整理成 (8*20, 512) 的形状
data = tensor.view(-1, 512).numpy()  # 转换为 NumPy 数组

# 创建标签
labels = np.repeat(np.arange(8), 20)  # 标签 [0,0,...,1,1,...,7,7,...] 共 160 个标签

# 应用 t-SNE 降维
tsne = TSNE(n_components=2, random_state=42)
reduced_data = tsne.fit_transform(data)

# 创建一个 DataFrame 以方便绘图
import pandas as pd
df = pd.DataFrame(reduced_data, columns=['Dimension 1', 'Dimension 2'])
df['Label'] = labels

# 使用 seaborn 绘图
plt.figure(figsize=(10, 8))
sns.scatterplot(data=df, x='Dimension 1', y='Dimension 2', hue='Label', palette='tab10', alpha=0.7)
plt.title('t-SNE Visualization of Tensor Data')
plt.legend(title='Label')
plt.savefig("result.png")