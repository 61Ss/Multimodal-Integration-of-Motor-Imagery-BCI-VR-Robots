import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch, butter, filtfilt
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr
import os
import warnings
import sys
from multiprocessing import Pool, cpu_count
from functools import partial
import time

# 解决Windows上KMeans的内存泄漏警告
os.environ['OMP_NUM_THREADS'] = '2'

# 导入现有的数据加载函数
from visualize_eeg_psd import load_eeg_data

class FastMutualInformationAnalyzer:
    """
    高性能EEG数据互信息分析器
    
    主要优化：
    1. 使用相关系数近似互信息（速度提升100x+）
    2. 只计算上三角矩阵（速度提升2x）
    3. 支持并行计算（速度提升4-8x）
    4. 数据预筛选和采样优化
    5. 内存优化的批处理
    """
    
    def __init__(self, subject_id='aw', random_state=42, use_incremental_pca=True, 
                 pca_batch_size=50, verbose=True, use_parallel=True, 
                 mi_method='correlation', max_samples_per_trial=1000):
        """
        初始化高性能互信息分析器
        
        Parameters:
        -----------
        subject_id : str
            受试者ID
        random_state : int
            随机种子
        use_incremental_pca : bool
            是否使用增量PCA
        pca_batch_size : int
            增量PCA的批处理大小
        verbose : bool
            是否显示详细输出信息
        use_parallel : bool
            是否使用并行计算
        mi_method : str
            互信息计算方法 ('correlation', 'mi_fast', 'mi_exact')
        max_samples_per_trial : int
            每个trial的最大采样点数（用于加速）
        """
        self.subject_id = subject_id
        self.random_state = random_state
        self.use_incremental_pca = use_incremental_pca
        self.verbose = verbose
        self.use_parallel = use_parallel
        self.mi_method = mi_method
        self.max_samples_per_trial = max_samples_per_trial
        
        # 频段定义 (Hz)
        self.bands = {
            'alpha': (7, 13),
            'beta': (14, 30), 
            'gamma': (30, 100)
        }
        
        # PCA和聚类参数
        self.pca_components = 20
        self.kmeans_clusters = 2
        self.n_neighbors = 5
        self.density_percentile = 85
        self.pca_batch_size = pca_batch_size
        
        # 并行计算参数
        self.n_jobs = min(cpu_count(), 8) if use_parallel else 1
    
    def fast_mutual_information(self, x, y, method='correlation'):
        """
        快速互信息计算
        
        Methods:
        - 'correlation': 使用皮尔逊相关系数平方作为互信息近似（最快）
        - 'mi_fast': 使用简化的MI估计（中等速度）
        - 'mi_exact': 使用精确的MI计算（最慢但最准确）
        """
        if method == 'correlation':
            # 使用相关系数的平方作为互信息的快速近似
            corr, _ = pearsonr(x, y)
            return corr ** 2
        
        elif method == 'mi_fast':
            # 简化的互信息估计（基于分桶）
            n_bins = min(20, len(x) // 10)
            hist_xy, x_edges, y_edges = np.histogram2d(x, y, bins=n_bins)
            hist_x = np.histogram(x, bins=x_edges)[0]
            hist_y = np.histogram(y, bins=y_edges)[0]
            
            # 避免log(0)
            hist_xy = hist_xy + 1e-10
            hist_x = hist_x + 1e-10
            hist_y = hist_y + 1e-10
            
            # 归一化
            p_xy = hist_xy / np.sum(hist_xy)
            p_x = hist_x / np.sum(hist_x)
            p_y = hist_y / np.sum(hist_y)
            
            # 计算互信息
            mi = 0
            for i in range(len(p_x)):
                for j in range(len(p_y)):
                    if p_xy[i, j] > 0:
                        mi += p_xy[i, j] * np.log(p_xy[i, j] / (p_x[i] * p_y[j]))
            
            return mi
        
        else:  # mi_exact
            # 使用sklearn的精确方法（最慢）
            from sklearn.feature_selection import mutual_info_regression
            return mutual_info_regression(
                x.reshape(-1, 1), y, 
                discrete_features=False, 
                n_neighbors=3,
                random_state=self.random_state
            )[0]
    
    def calculate_fast_mi_matrix(self, data):
        """
        快速计算互信息矩阵（只计算上三角）
        """
        num_channels = data.shape[1]
        
        # 数据采样以加速计算
        if data.shape[0] > self.max_samples_per_trial:
            indices = np.random.choice(data.shape[0], self.max_samples_per_trial, replace=False)
            data = data[indices]
        
        # 初始化矩阵
        mi_matrix = np.zeros((num_channels, num_channels))
        
        # 只计算上三角矩阵
        for i in range(num_channels):
            for j in range(i + 1, num_channels):
                mi_value = self.fast_mutual_information(
                    data[:, i], data[:, j], method=self.mi_method
                )
                mi_matrix[i, j] = mi_value
                mi_matrix[j, i] = mi_value  # 对称矩阵
        
        return mi_matrix
    
    def calculate_parallel_mi_matrix(self, data):
        """
        并行计算互信息矩阵
        """
        num_channels = data.shape[1]
        
        # 数据采样
        if data.shape[0] > self.max_samples_per_trial:
            indices = np.random.choice(data.shape[0], self.max_samples_per_trial, replace=False)
            data = data[indices]
        
        # 生成所有需要计算的通道对（只计算上三角）
        channel_pairs = [(i, j) for i in range(num_channels) for j in range(i + 1, num_channels)]
        
        # 创建部分函数
        mi_func = partial(self._compute_mi_pair, data=data)
        
        # 并行计算
        if self.n_jobs > 1:
            with Pool(self.n_jobs) as pool:
                mi_values = pool.map(mi_func, channel_pairs)
        else:
            mi_values = [mi_func(pair) for pair in channel_pairs]
        
        # 构建对称矩阵
        mi_matrix = np.zeros((num_channels, num_channels))
        for (i, j), mi_value in zip(channel_pairs, mi_values):
            mi_matrix[i, j] = mi_value
            mi_matrix[j, i] = mi_value
        
        return mi_matrix
    
    def _compute_mi_pair(self, channel_pair, data):
        """计算单个通道对的互信息（用于并行计算）"""
        i, j = channel_pair
        return self.fast_mutual_information(
            data[:, i], data[:, j], method=self.mi_method
        )
    
    def apply_bandpass_filter(self, data, band, fs):
        """应用带通滤波器"""
        fmin, fmax = band
        nyquist = fs / 2
        low = fmin / nyquist
        high = min(fmax / nyquist, 0.99)
        
        b, a = butter(4, [low, high], btype='band')
        
        filtered_data = np.zeros_like(data)
        for ch in range(data.shape[1]):
            filtered_data[:, ch] = filtfilt(b, a, data[:, ch])
            
        return filtered_data
    
    def analyze_single_trial_fast(self, trial_idx):
        """快速分析单个trial的互信息"""
        trial_data = self.trials[trial_idx]
        
        # 存储各频段的互信息矩阵
        band_mi_matrices = {}
        
        for band_name, band_range in self.bands.items():
            # 应用带通滤波
            filtered_data = self.apply_bandpass_filter(trial_data, band_range, self.fs)
            
            # 选择计算方法
            if self.use_parallel and self.n_jobs > 1:
                mi_matrix = self.calculate_parallel_mi_matrix(filtered_data)
            else:
                mi_matrix = self.calculate_fast_mi_matrix(filtered_data)
            
            band_mi_matrices[band_name] = mi_matrix
        
        return band_mi_matrices
    
    def load_and_preprocess_data(self):
        """加载并预处理EEG数据"""
        if self.verbose:
            print("===== 加载数据 =====")
        trials, fs, ch_names, ch_pos, trial_labels, regions = load_eeg_data(self.subject_id)
        
        self.trials = trials
        self.fs = fs
        self.ch_names = ch_names
        self.ch_pos = ch_pos
        self.trial_labels = trial_labels
        self.regions = regions
        
        num_trials, num_timepoints, num_channels = trials.shape
        if self.verbose:
            print(f"数据形状: {trials.shape}")
            print(f"计算复杂度: {num_channels*(num_channels-1)//2} 通道对 × 3 频段 × {num_trials} trials")
            print(f"使用方法: {self.mi_method}")
            print(f"并行计算: {'是' if self.use_parallel else '否'} ({self.n_jobs} cores)")
        
        return trials
    
    def calculate_density_labels(self):
        """计算密度标签（复用现有逻辑，简化输出）"""
        if self.verbose:
            print("===== 计算密度标签 =====")
        
        # 数据预处理：展平
        num_trials, num_timepoints, num_channels = self.trials.shape
        flattened_trials = self.trials.reshape(num_trials, -1)
        
        # PCA降维
        if self.use_incremental_pca:
            pca = IncrementalPCA(n_components=self.pca_components, batch_size=self.pca_batch_size)
            n_samples = flattened_trials.shape[0]
            for i in range(0, n_samples, self.pca_batch_size):
                batch = flattened_trials[i:i + self.pca_batch_size]
                pca.partial_fit(batch)
            pca_results = pca.transform(flattened_trials)
        else:
            pca = PCA(n_components=self.pca_components, random_state=self.random_state)
            pca_results = pca.fit_transform(flattened_trials)
        
        # K-means聚类
        kmeans = KMeans(n_clusters=self.kmeans_clusters, random_state=self.random_state)
        cluster_labels = kmeans.fit_predict(pca_results)
        
        # 计算KNN密度
        nn = NearestNeighbors(n_neighbors=self.n_neighbors)
        nn.fit(pca_results)
        distances, _ = nn.kneighbors(pca_results)
        density = np.mean(distances, axis=1)
        
        # 按类别划分密度阈值
        clusters = np.unique(cluster_labels)
        density_labels = [''] * num_trials
        
        for cluster in clusters:
            cluster_mask = cluster_labels == cluster
            cluster_density = density[cluster_mask]
            threshold = np.percentile(cluster_density, self.density_percentile)
            
            high_density_mask = cluster_mask & (density <= threshold)
            low_density_mask = cluster_mask & (density > threshold)
            
            for i in range(num_trials):
                if high_density_mask[i]:
                    density_labels[i] = 'high'
                elif low_density_mask[i]:
                    density_labels[i] = 'low'
        
        density_labels = np.array(density_labels, dtype='U10')
        self.density_labels = density_labels
        self.cluster_labels = cluster_labels
        
        if self.verbose:
            high_count = np.sum(density_labels == 'high')
            low_count = np.sum(density_labels == 'low')
            print(f"密度标签分布: High={high_count}, Low={low_count}")
        
        return density_labels
    
    def calculate_average_mi_and_rank_channels(self, band_mi_matrices):
        """计算平均互信息并排序通道"""
        # 计算三个频段的平均互信息矩阵
        alpha_mi = band_mi_matrices['alpha']
        beta_mi = band_mi_matrices['beta'] 
        gamma_mi = band_mi_matrices['gamma']
        
        avg_mi_matrix = (alpha_mi + beta_mi + gamma_mi) / 3
        
        # 计算每个通道的连接强度
        channel_connectivity = np.sum(avg_mi_matrix, axis=1)
        
        # 排序获取前4个通道
        top_channels_indices = np.argsort(channel_connectivity)[-4:][::-1]
        top_channels_scores = channel_connectivity[top_channels_indices]
        top_channels_names = [self.ch_names[i] for i in top_channels_indices]
        
        return avg_mi_matrix, top_channels_indices, top_channels_scores, top_channels_names
    
    def analyze_all_trials_fast(self):
        """快速分析所有trials"""
        if self.verbose:
            print("===== 开始快速互信息分析 =====")
        
        if not hasattr(self, 'density_labels'):
            self.calculate_density_labels()
        
        num_trials = self.trials.shape[0]
        results = []
        
        start_time = time.time()
        
        for trial_idx in range(num_trials):
            # 进度显示
            if self.verbose and trial_idx % max(1, num_trials // 10) == 0:
                elapsed = time.time() - start_time
                estimated_total = elapsed * num_trials / (trial_idx + 1) if trial_idx > 0 else 0
                remaining = estimated_total - elapsed
                print(f"进度: {trial_idx+1}/{num_trials} ({100*trial_idx/num_trials:.0f}%) - "
                      f"用时: {elapsed:.1f}s, 预计剩余: {remaining:.1f}s")
            
            # 分析当前trial
            band_mi_matrices = self.analyze_single_trial_fast(trial_idx)
            avg_mi_matrix, top_indices, top_scores, top_names = self.calculate_average_mi_and_rank_channels(band_mi_matrices)
            
            # 获取trial信息
            original_label = self.trial_labels[trial_idx]
            density_label = self.density_labels[trial_idx]
            cluster_label = self.cluster_labels[trial_idx]
            
            # 保存结果
            trial_result = {
                'trial_idx': trial_idx,
                'original_label': original_label,
                'density_label': density_label,
                'cluster_label': cluster_label,
                'top_4_channels': top_names,
                'top_4_indices': top_indices,
                'top_4_scores': top_scores,
                'avg_mi_matrix': avg_mi_matrix,
                'band_mi_matrices': band_mi_matrices
            }
            
            results.append(trial_result)
        
        total_time = time.time() - start_time
        if self.verbose:
            print(f"分析完成! 总用时: {total_time:.1f}秒 ({total_time/num_trials:.2f}秒/trial)")
        
        self.results = results
        return results
    
    def save_results_to_csv(self, output_path='fast_mutual_information_results.csv'):
        """保存结果到CSV"""
        if self.verbose:
            print("===== 保存结果到CSV =====")
        
        data_rows = []
        for result in self.results:
            row = {
                'trial_idx': result['trial_idx'],
                'original_label': result['original_label'],
                'density_label': result['density_label'],
                'cluster_label': result['cluster_label'],
                'top_channel_1': result['top_4_channels'][0],
                'top_channel_1_score': result['top_4_scores'][0],
                'top_channel_2': result['top_4_channels'][1],
                'top_channel_2_score': result['top_4_scores'][1],
                'top_channel_3': result['top_4_channels'][2],
                'top_channel_3_score': result['top_4_scores'][2],
                'top_channel_4': result['top_4_channels'][3],
                'top_channel_4_score': result['top_4_scores'][3],
            }
            data_rows.append(row)
        
        df = pd.DataFrame(data_rows)
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        if self.verbose:
            print(f"结果已保存至: {output_path}")
        
        return df
    
    def run_fast_analysis(self, max_trials=None):
        """运行快速分析流程"""
        if self.verbose:
            print("===== 开始高性能互信息分析 =====")
            print(f"优化策略: {self.mi_method} + {'并行计算' if self.use_parallel else '串行计算'}")
        
        # 1. 加载数据
        self.load_and_preprocess_data()
        
        # 2. 计算密度标签
        self.calculate_density_labels()
        
        # 3. 截取数据（如果需要）
        if max_trials is not None and max_trials < self.trials.shape[0]:
            if self.verbose:
                print(f"测试模式：只分析前 {max_trials} 个trials")
            self.trials = self.trials[:max_trials]
            self.trial_labels = self.trial_labels[:max_trials]
            self.density_labels = self.density_labels[:max_trials]
            self.cluster_labels = self.cluster_labels[:max_trials]
        
        # 4. 快速分析
        self.analyze_all_trials_fast()
        
        # 5. 保存结果
        suffix = f'_first{max_trials}' if max_trials is not None else ''
        output_csv = f'fast_mutual_information_results_{self.subject_id}{suffix}.csv'
        df = self.save_results_to_csv(output_csv)
        
        if self.verbose:
            print("===== 高性能分析完成 =====")
        
        return self.results, df

def main():
    """主函数"""
    # 解析命令行参数
    quick_test = '--quick' in sys.argv or '-q' in sys.argv
    silent_mode = '--silent' in sys.argv or '-s' in sys.argv
    use_exact_mi = '--exact' in sys.argv
    disable_parallel = '--no-parallel' in sys.argv
    
    max_trials = 5 if quick_test else None
    
    # 检查自定义trials数量
    for arg in sys.argv:
        if arg.startswith('--trials='):
            try:
                max_trials = int(arg.split('=')[1])
                quick_test = True
            except ValueError:
                if not silent_mode:
                    print("⚠️ 无效的trials参数")
    
    # 选择互信息计算方法
    if use_exact_mi:
        mi_method = 'mi_exact'
        if not silent_mode:
            print("使用精确互信息计算（较慢但最准确）")
    else:
        mi_method = 'correlation'
        if not silent_mode:
            print("使用相关系数近似互信息（快速模式）")
    
    # 创建高性能分析器
    analyzer = FastMutualInformationAnalyzer(
        subject_id='aw',
        random_state=42,
        use_incremental_pca=True,
        pca_batch_size=50,
        verbose=not silent_mode,
        use_parallel=not disable_parallel,
        mi_method=mi_method,
        max_samples_per_trial=500  # 减少采样点以进一步加速
    )
    
    # 显示模式信息
    if not silent_mode:
        print("开始高性能分析...")
        if quick_test:
            print(f"🚀 快速测试模式：{max_trials} trials")
        else:
            print("📊 完整分析模式")
        print(f"方法: {mi_method}")
        print(f"并行: {'是' if not disable_parallel else '否'}")
        print("命令行选项:")
        print("  --quick: 快速测试")
        print("  --silent: 静默模式")
        print("  --exact: 使用精确MI计算")
        print("  --no-parallel: 禁用并行计算")
        print("  --trials=N: 自定义trials数量")
    
    # 运行分析
    start_time = time.time()
    results, df = analyzer.run_fast_analysis(max_trials=max_trials)
    total_time = time.time() - start_time
    
    # 显示结果
    if not silent_mode:
        print(f"\n===== 性能统计 =====")
        print(f"总耗时: {total_time:.1f}秒")
        print(f"平均每trial: {total_time/len(results):.2f}秒")
        print(f"处理的trials: {len(results)}")
        
        print(f"\n===== 样例结果 =====")
        for i in range(min(3, len(results))):
            result = results[i]
            print(f"Trial {result['trial_idx']}:")
            print(f"  原始标签: {result['original_label']}")
            print(f"  密度标签: {result['density_label']}")
            print(f"  前4通道: {result['top_4_channels']}")
            print()
    else:
        print(f"完成: {len(results)} trials, {total_time:.1f}s")

if __name__ == '__main__':
    main()