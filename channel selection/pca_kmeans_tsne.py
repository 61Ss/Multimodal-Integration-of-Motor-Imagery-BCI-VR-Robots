import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.cm as cm

# 导入数据加载函数（复用现有逻辑）
from visualize_eeg_psd import load_eeg_data

def main():
    # ---------------------- 配置参数 ----------------------
    SUBJECT_ID = 'aw'  # 被试ID
    PCA_COMPONENTS = 130  # PCA降维目标维度
    KMEANS_CLUSTERS = 2  # K-means聚类数量
    RANDOM_STATE = 42  # 随机种子，保证结果可复现

    # ---------------------- 加载与预处理数据 ----------------------
    print("===== 加载数据 =====")
    trials, fs, ch_names, ch_pos, trial_labels, regions = load_eeg_data(SUBJECT_ID)
    num_trials, num_timepoints, num_channels = trials.shape
    print(f"原始数据形状: {trials.shape} (trials, timepoints, channels)")

    # 将每个trial展平为一维向量 (num_trials, num_timepoints*num_channels)
    flattened_trials = trials.reshape(num_trials, -1)
    print(f"展平后数据形状: {flattened_trials.shape} (trials, features)")

    # ---------------------- PCA降维 ----------------------
    print("\n===== 执行PCA降维 =====")
    pca = PCA(n_components=PCA_COMPONENTS, random_state=RANDOM_STATE)
    pca_results = pca.fit_transform(flattened_trials)
    print(f"PCA降维后形状: {pca_results.shape} (trials, {PCA_COMPONENTS} components)")
    print(f"解释方差比例: {np.sum(pca.explained_variance_ratio_):.2%}")

    # ---------------------- K-means聚类 ----------------------
    print("\n===== 执行K-means聚类 =====")
    kmeans = KMeans(n_clusters=KMEANS_CLUSTERS, random_state=RANDOM_STATE)
    cluster_labels = kmeans.fit_predict(pca_results)
    print(f"聚类完成，标签形状: {cluster_labels.shape} (trials,)")
    print(f"聚类中心形状: {kmeans.cluster_centers_.shape}")

    # ---------------------- t-SNE可视化 ----------------------
    print("\n===== 执行t-SNE可视化 =====")
    tsne = TSNE(n_components=2, random_state=RANDOM_STATE, perplexity=min(30, num_trials-1))
    tsne_results = tsne.fit_transform(pca_results)
    print(f"t-SNE降维后形状: {tsne_results.shape} (trials, 2 components)")

    # ---------------------- 新增：计算KNN密度 ----------------------
    from sklearn.neighbors import NearestNeighbors
    N_NEIGHBORS = 5  # KNN邻居数量
    nn = NearestNeighbors(n_neighbors=N_NEIGHBORS)
    nn.fit(pca_results)  # 使用PCA特征计算距离
    distances, _ = nn.kneighbors(pca_results)
    density = np.mean(distances, axis=1)  # 平均距离越低，密度越高

    # ---------------------- 按类别划分密度阈值 ----------------------
    clusters = np.unique(cluster_labels)
    density_thresholds = {}
    for cluster in clusters:
        # 计算每个类内的密度阈值（使用中位数）
        cluster_density = density[cluster_labels == cluster]
        # density_thresholds[cluster] = np.median(cluster_density)
        density_thresholds[cluster] = np.percentile(cluster_density, 85)  # 使用25%分位数

    # ---------------------- 绘制聚类结果（区分密度） ----------------------
    plt.figure(figsize=(12, 8))
    # 修复1：使用新的colormap获取方式（替换cm.get_cmap）
    cmap = plt.colormaps['viridis'].resampled(KMEANS_CLUSTERS)

    # 创建用于colorbar的虚拟散点
    dummy_scatter = plt.scatter([], [], c=[], cmap=cmap, label='Cluster')

    for cluster in clusters:
        # 获取当前类别的所有点
        mask = cluster_labels == cluster
        cluster_tsne = tsne_results[mask]
        cluster_density = density[mask]
        threshold = density_thresholds[cluster]

        # 区分高低密度点
        high_density_mask = cluster_density <= threshold  # 低密度=平均距离大
        low_density_mask = cluster_density > threshold

        # 绘制高密度点（圆形）
        plt.scatter(
            cluster_tsne[high_density_mask, 0],
            cluster_tsne[high_density_mask, 1],
            marker='o',  # 圆形表示高密度
            color=cmap(cluster),
            alpha=0.8,
            s=50,
            label=f'Cluster {cluster} (High Density)'
        )

        # 绘制低密度点（三角形）
        plt.scatter(
            cluster_tsne[low_density_mask, 0],
            cluster_tsne[low_density_mask, 1],
            marker='^',  # 三角形表示低密度
            color=cmap(cluster),
            alpha=0.8,
            s=50,
            label=f'Cluster {cluster} (Low Density)'
        )

    # 修复2：使用虚拟散点创建colorbar
    plt.colorbar(dummy_scatter, ticks=range(KMEANS_CLUSTERS), label='Cluster Label')
    plt.title(f't-SNE Visualization with Density-based Shapes (k={KMEANS_CLUSTERS})', fontsize=14)
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend()  # 添加图例区分高低密度
    plt.grid(alpha=0.3)

    # 保存可视化结果
    output_path = f'tsne_cluster_density_visualization_{SUBJECT_ID}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"密度可视化结果已保存至: {output_path}")
    plt.close()

    print("\n===== 流程完成 =====")

if __name__ == '__main__':
    main()