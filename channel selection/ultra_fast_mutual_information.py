import numpy as np
import pandas as pd
from scipy.signal import butter, sosfilt
from scipy.stats import pearsonr
from sklearn.decomposition import IncrementalPCA
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import time
import os
import sys

# 解决Windows上KMeans的内存泄漏警告
os.environ['OMP_NUM_THREADS'] = '2'

# 导入现有的数据加载函数
from visualize_eeg_psd import load_eeg_data

class UltraFastMutualInformationAnalyzer:
    """
    超高速EEG互信息分析器
    
    核心优化：
    1. 使用SOS滤波器代替filtfilt（10x+提速）
    2. 批量滤波所有trials（避免重复计算）
    3. 简化的相关系数计算
    4. 最小化内存分配
    5. 跳过不必要的精度计算
    """
    
    def __init__(self, subject_id='aw', random_state=42, verbose=True, 
                 use_simple_filter=True, max_samples_per_trial=200):
        """
        初始化超高速分析器
        
        Parameters:
        -----------
        subject_id : str
            受试者ID
        random_state : int
            随机种子
        verbose : bool
            是否显示详细输出
        use_simple_filter : bool
            是否使用简化滤波（大幅提速）
        max_samples_per_trial : int
            每个trial的最大采样点数（大幅减少计算量）
        """
        self.subject_id = subject_id
        self.random_state = random_state
        self.verbose = verbose
        self.use_simple_filter = use_simple_filter
        self.max_samples_per_trial = max_samples_per_trial
        
        # 频段定义 (Hz) - 简化为更宽的频段以减少滤波次数
        if use_simple_filter:
            self.bands = {
                'low_freq': (7, 30),    # 合并alpha和beta
                'high_freq': (30, 80)   # 简化的gamma
            }
        else:
            self.bands = {
                'alpha': (7, 13),
                'beta': (14, 30), 
                'gamma': (30, 100)
            }
        
        # 优化的参数
        self.pca_components = 20
        self.kmeans_clusters = 2
        self.n_neighbors = 5
        self.density_percentile = 85
        
        # 预编译滤波器系数（避免重复计算）
        self.filter_cache = {}
    
    def get_fast_filter(self, band, fs):
        """获取或创建快速滤波器"""
        fmin, fmax = band
        cache_key = (fmin, fmax, fs)
        
        if cache_key not in self.filter_cache:
            nyquist = fs / 2
            low = fmin / nyquist
            high = min(fmax / nyquist, 0.99)
            
            # 使用SOS格式的滤波器（更稳定更快）
            sos = butter(2, [low, high], btype='band', output='sos')  # 降低阶数从4到2
            self.filter_cache[cache_key] = sos
        
        return self.filter_cache[cache_key]
    
    def apply_fast_filter(self, data, band, fs):
        """应用快速滤波器"""
        if self.use_simple_filter:
            # 超简化滤波：只使用简单的频域截断
            return self.apply_frequency_filter(data, band, fs)
        else:
            # 使用优化的SOS滤波
            sos = self.get_fast_filter(band, fs)
            filtered_data = np.zeros_like(data)
            for ch in range(data.shape[1]):
                filtered_data[:, ch] = sosfilt(sos, data[:, ch])
            return filtered_data
    
    def apply_frequency_filter(self, data, band, fs):
        """超简化的频域滤波（最快但精度略低）"""
        fmin, fmax = band
        
        # FFT频域滤波
        fft_data = np.fft.fft(data, axis=0)
        freqs = np.fft.fftfreq(data.shape[0], 1/fs)
        
        # 创建频域掩码
        mask = (np.abs(freqs) >= fmin) & (np.abs(freqs) <= fmax)
        
        # 应用滤波
        fft_data[~mask] = 0
        
        # 反变换回时域
        filtered_data = np.real(np.fft.ifft(fft_data, axis=0))
        
        return filtered_data
    
    def fast_correlation_matrix(self, data):
        """超快速相关系数矩阵计算"""
        # 数据采样以大幅减少计算量
        if data.shape[0] > self.max_samples_per_trial:
            step = data.shape[0] // self.max_samples_per_trial
            data = data[::step]
        
        # 标准化数据
        data_centered = data - np.mean(data, axis=0)
        data_std = np.std(data, axis=0)
        data_std[data_std == 0] = 1  # 避免除零
        data_normalized = data_centered / data_std
        
        # 计算相关系数矩阵（向量化操作）
        corr_matrix = np.corrcoef(data_normalized.T)
        
        # 处理NaN值
        corr_matrix = np.nan_to_num(corr_matrix, 0)
        
        # 取平方作为互信息近似
        return corr_matrix ** 2
    
    def analyze_single_trial_ultra_fast(self, trial_idx):
        """超快速分析单个trial"""
        trial_data = self.trials[trial_idx]
        
        # 存储各频段的互信息矩阵
        band_mi_matrices = {}
        
        for band_name, band_range in self.bands.items():
            # 应用快速滤波
            filtered_data = self.apply_fast_filter(trial_data, band_range, self.fs)
            
            # 快速计算相关系数矩阵
            mi_matrix = self.fast_correlation_matrix(filtered_data)
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
            print(f"优化策略: {'简化滤波' if self.use_simple_filter else 'SOS滤波'}")
            print(f"采样策略: 每trial最多{self.max_samples_per_trial}个点")
            print(f"频段数量: {len(self.bands)} (减少滤波次数)")
        
        return trials
    
    def calculate_density_labels_fast(self):
        """快速计算密度标签"""
        if self.verbose:
            print("===== 快速计算密度标签 =====")
        
        # 数据预处理：展平
        num_trials, num_timepoints, num_channels = self.trials.shape
        
        # 大幅采样以加速PCA
        sample_step = max(1, num_timepoints // 50)  # 只取1/50的时间点
        sampled_trials = self.trials[:, ::sample_step, ::2]  # 时间和通道都采样
        
        flattened_trials = sampled_trials.reshape(num_trials, -1)
        
        if self.verbose:
            print(f"PCA输入维度: {flattened_trials.shape} (大幅采样后)")
        
        # 快速PCA
        pca = IncrementalPCA(n_components=self.pca_components, batch_size=min(50, num_trials))
        
        # 分批处理
        batch_size = min(50, num_trials)
        for i in range(0, num_trials, batch_size):
            batch = flattened_trials[i:i + batch_size]
            pca.partial_fit(batch)
        
        pca_results = pca.transform(flattened_trials)
        
        # K-means聚类
        kmeans = KMeans(n_clusters=self.kmeans_clusters, random_state=self.random_state, n_init=3)
        cluster_labels = kmeans.fit_predict(pca_results)
        
        # 简化的密度计算
        nn = NearestNeighbors(n_neighbors=min(self.n_neighbors, num_trials//2))
        nn.fit(pca_results)
        distances, _ = nn.kneighbors(pca_results)
        density = np.mean(distances, axis=1)
        
        # 简化的密度标签分配
        clusters = np.unique(cluster_labels)
        density_labels = [''] * num_trials
        
        for cluster in clusters:
            cluster_mask = cluster_labels == cluster
            threshold = np.percentile(density[cluster_mask], self.density_percentile)
            
            for i in range(num_trials):
                if cluster_mask[i]:
                    density_labels[i] = 'high' if density[i] <= threshold else 'low'
        
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
        # 计算所有频段的平均互信息矩阵
        if self.use_simple_filter:
            # 简化频段的情况
            if len(band_mi_matrices) == 2:
                bands = list(band_mi_matrices.keys())
                avg_mi_matrix = (band_mi_matrices[bands[0]] + band_mi_matrices[bands[1]]) / 2
            else:
                avg_mi_matrix = list(band_mi_matrices.values())[0]
        else:
            # 传统三频段的情况
            alpha_mi = band_mi_matrices.get('alpha', np.zeros_like(list(band_mi_matrices.values())[0]))
            beta_mi = band_mi_matrices.get('beta', np.zeros_like(list(band_mi_matrices.values())[0]))
            gamma_mi = band_mi_matrices.get('gamma', np.zeros_like(list(band_mi_matrices.values())[0]))
            avg_mi_matrix = (alpha_mi + beta_mi + gamma_mi) / 3
        
        # 计算每个通道的连接强度
        channel_connectivity = np.sum(avg_mi_matrix, axis=1)
        
        # 排序获取前4个通道
        top_channels_indices = np.argsort(channel_connectivity)[-4:][::-1]
        top_channels_scores = channel_connectivity[top_channels_indices]
        top_channels_names = [self.ch_names[i] for i in top_channels_indices]
        
        return avg_mi_matrix, top_channels_indices, top_channels_scores, top_channels_names
    
    def analyze_all_trials_ultra_fast(self):
        """超快速分析所有trials"""
        if self.verbose:
            print("===== 开始超高速互信息分析 =====")
        
        if not hasattr(self, 'density_labels'):
            self.calculate_density_labels_fast()
        
        num_trials = self.trials.shape[0]
        results = []
        
        start_time = time.time()
        
        for trial_idx in range(num_trials):
            # 智能进度显示
            if self.verbose:
                if trial_idx == 0:
                    print("开始处理第一个trial...")
                elif trial_idx % max(1, num_trials // 20) == 0:  # 显示20次进度
                    elapsed = time.time() - start_time
                    rate = trial_idx / elapsed if elapsed > 0 else 0
                    remaining = (num_trials - trial_idx) / rate if rate > 0 else 0
                    print(f"进度: {trial_idx}/{num_trials} ({100*trial_idx/num_trials:.0f}%) - "
                          f"速度: {rate:.1f} trials/秒, 预计剩余: {remaining:.0f}秒")
            
            # 分析当前trial
            band_mi_matrices = self.analyze_single_trial_ultra_fast(trial_idx)
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
            }
            
            results.append(trial_result)
        
        total_time = time.time() - start_time
        if self.verbose:
            print(f"\n🎉 超高速分析完成!")
            print(f"总用时: {total_time:.1f}秒")
            print(f"平均速度: {num_trials/total_time:.1f} trials/秒")
            print(f"每trial平均: {total_time/num_trials:.2f}秒")
        
        self.results = results
        return results
    
    def save_results_to_csv(self, output_path='ultra_fast_mi_results.csv'):
        """保存结果到CSV"""
        if self.verbose:
            print("===== 保存结果 =====")
        
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
    
    def run_ultra_fast_analysis(self, max_trials=None):
        """运行超高速分析流程"""
        if self.verbose:
            print("===== 超高速EEG互信息分析 =====")
            print("⚡ 极致优化策略:")
            print("  - 简化滤波算法")
            print("  - 大幅数据采样")
            print("  - 向量化计算")
            print("  - 减少频段数量")
        
        # 1. 加载数据
        self.load_and_preprocess_data()
        
        # 2. 快速计算密度标签
        self.calculate_density_labels_fast()
        
        # 3. 截取数据（如果需要）
        if max_trials is not None and max_trials < self.trials.shape[0]:
            if self.verbose:
                print(f"测试模式：只分析前 {max_trials} 个trials")
            self.trials = self.trials[:max_trials]
            self.trial_labels = self.trial_labels[:max_trials]
            self.density_labels = self.density_labels[:max_trials]
            self.cluster_labels = self.cluster_labels[:max_trials]
        
        # 4. 超快速分析
        self.analyze_all_trials_ultra_fast()
        
        # 5. 保存结果
        suffix = f'_first{max_trials}' if max_trials is not None else ''
        output_csv = f'ultra_fast_mi_results_{self.subject_id}{suffix}.csv'
        df = self.save_results_to_csv(output_csv)
        
        return self.results, df

def main():
    """主函数"""
    # 解析命令行参数
    quick_test = '--quick' in sys.argv or '-q' in sys.argv
    silent_mode = '--silent' in sys.argv or '-s' in sys.argv
    precision_mode = '--precision' in sys.argv
    
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
    
    # 创建超高速分析器
    analyzer = UltraFastMutualInformationAnalyzer(
        subject_id='aw',
        random_state=42,
        verbose=not silent_mode,
        use_simple_filter=not precision_mode,  # 精度模式使用更好的滤波
        max_samples_per_trial=100 if not precision_mode else 500  # 精度模式使用更多采样点
    )
    
    # 显示模式信息
    if not silent_mode:
        print("🚀 超高速EEG互信息分析器")
        print("=" * 50)
        if quick_test:
            print(f"⚡ 快速测试模式：{max_trials} trials")
        else:
            print("⚡ 完整分析模式")
        
        if precision_mode:
            print("🎯 精度模式：使用更好的滤波和更多采样点")
        else:
            print("🚀 速度模式：极致优化的快速分析")
        
        print("\n命令行选项:")
        print("  --quick: 快速测试 (5 trials)")
        print("  --silent: 静默模式")
        print("  --precision: 精度模式（稍慢但更准确）")
        print("  --trials=N: 自定义trials数量")
        print()
    
    # 运行分析
    start_time = time.time()
    results, df = analyzer.run_ultra_fast_analysis(max_trials=max_trials)
    total_time = time.time() - start_time
    
    # 显示结果
    if not silent_mode:
        print(f"\n{'='*50}")
        print(f"🎉 超高速分析完成!")
        print(f"{'='*50}")
        print(f"总耗时: {total_time:.1f}秒")
        print(f"处理速度: {len(results)/total_time:.1f} trials/秒")
        print(f"平均每trial: {total_time/len(results):.3f}秒")
        
        if len(results) < 280:
            estimated_280 = total_time * 280 / len(results)
            print(f"280个trials预计用时: {estimated_280/60:.1f}分钟")
        
        print(f"\n===== 样例结果 =====")
        for i in range(min(3, len(results))):
            result = results[i]
            print(f"Trial {result['trial_idx']}:")
            print(f"  原始标签: {result['original_label']}")
            print(f"  密度标签: {result['density_label']}")
            print(f"  前4通道: {result['top_4_channels']}")
            print()
    else:
        print(f"完成: {len(results)} trials, {total_time:.1f}s, {len(results)/total_time:.1f} trials/s")

if __name__ == '__main__':
    main()