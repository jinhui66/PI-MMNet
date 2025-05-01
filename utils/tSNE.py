import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def visualize_tsne(data, labels=None, perplexity=5, path='tSNE.png'):
    """
    使用t-SNE降维并可视化数据
    
    参数:
        data: numpy数组，shape为(n_samples, n_features)
        labels: 可选，每个样本的标签，用于着色
        perplexity: t-SNE的困惑度参数
        title: 图标题
    """
    # 创建t-SNE模型
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    
    # 执行降维
    tsne_results = tsne.fit_transform(data)
    
    # 可视化
    plt.figure(figsize=(10, 8))
    
    if labels is not None:
        scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels, cmap='viridis', alpha=0.6)
        plt.colorbar(scatter)
    else:
        plt.scatter(tsne_results[:, 0], tsne_results[:, 1], alpha=0.6)
    
    # plt.title(title)
    # plt.xlabel('t-SNE 1')
    # plt.ylabel('t-SNE 2')
    # plt.grid(True)
    plt.savefig(path)
    
    return tsne_results

def draw_tsne(data, labels=None, perplexity=5, path='tSNE.png'):
    """
    绘制t-SNE图像并保存为PNG文件
    """
    data = data.detach().cpu().numpy()  # 将数据从GPU转移到CPU并转换为numpy数组
    labels = labels.detach().cpu().numpy() if labels is not None else None 
    tsne_results = visualize_tsne(data, labels, perplexity, path)

if __name__ == "__main__":
    # 假设你的数据是一个numpy数组，shape为(batch_size, 128)
    data = np.random.randn(10, 128)  # 这里用随机数据示例，替换为你的实际数据
    labels = np.random.randint(0, 2, size=(10,))  # 随机标签，替换为你的实际标签
    tsne_results = visualize_tsne(data,labels)