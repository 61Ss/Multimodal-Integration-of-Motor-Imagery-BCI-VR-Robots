#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
精确互信息分析 - 使用真实的信息论互信息计算

⚠️ 警告：此版本速度极慢但结果精确，适合小样本或最终验证
"""

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfilt
from sklearn.feature_selection import mutual_info_regression
from sklearn.decomposition import IncrementalPCA
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import time
import os
import sys
from functools import partial
from multiprocessing import Pool

# 解决Windows上KMeans的内存泄漏警告
os.environ['OMP_NUM_THREADS'] = '2'

# 导入现有的数据加载函数
from visualize_eeg_psd import load_eeg_data

class ExactMutualInformationAnalyzer:
    """
    精确EEG互信息分析器 - 使用真实的信息论互信息
    
    适用于：
    1. 小样本精确分析 (≤20 trials)
    2. 方法验证
    3. 最终结果确认
    """
    
    def __init__(self, subject_id='aw', random_state=42, verbose=True, 
                 n_jobs=4, mi_neighbors=3):
        """
        初始化精确分析器
        
        Parameters:
        -----------
        n_jobs : int
            并行计算的进程数
        mi_neighbors : int
            互信息计算中的邻居数（影响精度和速度）
        """
        self.subject_id = subject_id
        self.random_state = random_state
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.mi_neighbors = mi_neighbors
        
        # 频段定义
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
    
    def calculate_exact_mutual_information_matrix(self, data):
        """
        计算精确的互信息矩阵
        
        使用sklearn的mutual_info_regression计算真实的互信息
        """
        num_channels = data.shape[1]
        mi_matrix = np.zeros((num_channels, num_channels))
        
        if self.verbose:
            print(f"    计算{num_channels}×{num_channels}精确互信息矩阵...")
        
        # 计算所有通道对的互信息
        channel_pairs = [(i, j) for i in range(num_channels) for j in range(i + 1, num_channels)]
        total_pairs = len(channel_pairs)
        
        if self.verbose:
            print(f"    总共{total_pairs}个通道对需要计算")
        
        if self.n_jobs > 1:
            # 并行计算
            mi_func = partial(self._compute_exact_mi_pair, data=data)
            with Pool(self.n_jobs) as pool:
                mi_values = pool.map(mi_func, channel_pairs)
        else:
            # 串行计算
            mi_values = []
            for idx, (i, j) in enumerate(channel_pairs):
                if self.verbose and idx % max(1, total_pairs // 10) == 0:
                    print(f"      进度: {idx}/{total_pairs} ({100*idx/total_pairs:.0f}%)")
                
                mi_value = mutual_info_regression(
                    data[:, i].reshape(-1, 1), 
                    data[:, j],
                    discrete_features=False,
                    n_neighbors=self.mi_neighbors,
                    random_state=self.random_state
                )[0]
                mi_values.append(mi_value)
        
        # 填充对称矩阵
        for (i, j), mi_value in zip(channel_pairs, mi_values):
            mi_matrix[i, j] = mi_value
            mi_matrix[j, i] = mi_value
        
        # 对角线设为0（自己与自己的互信息不考虑）
        np.fill_diagonal(mi_matrix, 0)
        
        return mi_matrix
    
    def _compute_exact_mi_pair(self, channel_pair, data):
        """计算单个通道对的精确互信息（用于并行处理）"""
        i, j = channel_pair
        mi_value = mutual_info_regression(
            data[:, i].reshape(-1, 1), 
            data[:, j],
            discrete_features=False,
            n_neighbors=self.mi_neighbors,
            random_state=self.random_state
        )[0]
        return mi_value
    
    def apply_bandpass_filter(self, data, band, fs):
        """应用带通滤波"""
        fmin, fmax = band
        nyquist = fs / 2
        low = fmin / nyquist
        high = min(fmax / nyquist, 0.99)
        
        # 使用SOS格式滤波器
        sos = butter(2, [low, high], btype='band', output='sos')
        
        filtered_data = np.zeros_like(data)
        for ch in range(data.shape[1]):
            filtered_data[:, ch] = sosfilt(sos, data[:, ch])
        
        return filtered_data
    
    def analyze_single_trial_exact(self, trial_idx):
        """精确分析单个trial"""
        if self.verbose:
            print(f"\n分析Trial {trial_idx}...")
        
        trial_data = self.trials[trial_idx]
        
        # 存储各频段的互信息矩阵
        band_mi_matrices = {}
        
        for band_name, band_range in self.bands.items():
            if self.verbose:
                print(f"  处理{band_name}频段 ({band_range[0]}-{band_range[1]} Hz)...")
            
            # 滤波
            start_time = time.time()
            filtered_data = self.apply_bandpass_filter(trial_data, band_range, self.fs)
            filter_time = time.time() - start_time
            
            if self.verbose:
                print(f"    滤波耗时: {filter_time:.2f}秒")
            
            # 计算精确互信息矩阵
            start_time = time.time()
            mi_matrix = self.calculate_exact_mutual_information_matrix(filtered_data)
            mi_time = time.time() - start_time
            
            if self.verbose:
                print(f"    互信息计算耗时: {mi_time:.2f}秒")
                print(f"    互信息矩阵统计: 均值={np.mean(mi_matrix):.4f}, 最大值={np.max(mi_matrix):.4f}")
            
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
        
        if self.verbose:
            print(f"数据形状: {trials.shape}")
            print(f"使用精确互信息计算（sklearn.mutual_info_regression）")
            print(f"并行进程数: {self.n_jobs}")
            print(f"互信息邻居数: {self.mi_neighbors}")
        
        return trials
    
    def calculate_density_labels_fast(self):
        """快速计算密度标签（与ultra_fast版本相同）"""
        if self.verbose:
            print("===== 计算密度标签 =====")
        
        # 使用采样版本快速计算密度标签
        num_trials, num_timepoints, num_channels = self.trials.shape
        
        sample_step = max(1, num_timepoints // 50)
        sampled_trials = self.trials[:, ::sample_step, ::2]
        flattened_trials = sampled_trials.reshape(num_trials, -1)
        
        # 快速PCA
        pca = IncrementalPCA(n_components=self.pca_components, batch_size=min(50, num_trials))
        
        batch_size = min(50, num_trials)
        for i in range(0, num_trials, batch_size):
            batch = flattened_trials[i:i + batch_size]
            pca.partial_fit(batch)
        
        pca_results = pca.transform(flattened_trials)
        
        # K-means聚类
        kmeans = KMeans(n_clusters=self.kmeans_clusters, random_state=self.random_state, n_init=3)
        cluster_labels = kmeans.fit_predict(pca_results)
        
        # 密度计算
        nn = NearestNeighbors(n_neighbors=min(self.n_neighbors, num_trials//2))
        nn.fit(pca_results)
        distances, _ = nn.kneighbors(pca_results)
        density = np.mean(distances, axis=1)
        
        # 密度标签分配
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
        # 计算三个频段的平均
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
    
    def run_exact_analysis(self, max_trials=None):
        """运行精确分析流程"""
        if self.verbose:
            print("===== 精确EEG互信息分析 =====")
            print("🎯 使用真实的信息论互信息计算")
            print("⚠️  警告：速度极慢，建议只用于小样本")
        
        # 加载数据
        self.load_and_preprocess_data()
        
        # 计算密度标签
        self.calculate_density_labels_fast()
        
        # 限制trials数量
        if max_trials is not None and max_trials < self.trials.shape[0]:
            if self.verbose:
                print(f"限制分析：只处理前 {max_trials} 个trials")
            self.trials = self.trials[:max_trials]
            self.trial_labels = self.trial_labels[:max_trials]
            self.density_labels = self.density_labels[:max_trials]
            self.cluster_labels = self.cluster_labels[:max_trials]
        
        # 分析所有trials
        num_trials = self.trials.shape[0]
        results = []
        
        overall_start = time.time()
        
        for trial_idx in range(num_trials):
            trial_start = time.time()
            
            # 分析当前trial
            band_mi_matrices = self.analyze_single_trial_exact(trial_idx)
            avg_mi_matrix, top_indices, top_scores, top_names = self.calculate_average_mi_and_rank_channels(band_mi_matrices)
            
            # 保存结果
            trial_result = {
                'trial_idx': trial_idx,
                'original_label': self.trial_labels[trial_idx],
                'density_label': self.density_labels[trial_idx],
                'cluster_label': self.cluster_labels[trial_idx],
                'top_4_channels': top_names,
                'top_4_indices': top_indices,
                'top_4_scores': top_scores,
            }
            
            results.append(trial_result)
            
            trial_time = time.time() - trial_start
            remaining_trials = num_trials - trial_idx - 1
            estimated_remaining = trial_time * remaining_trials
            
            if self.verbose:
                print(f"\nTrial {trial_idx} 完成 (耗时: {trial_time:.1f}秒)")
                print(f"预计剩余时间: {estimated_remaining/60:.1f}分钟")
        
        total_time = time.time() - overall_start
        
        if self.verbose:
            print(f"\n🎉 精确分析完成!")
            print(f"总用时: {total_time/60:.1f}分钟")
            print(f"平均每trial: {total_time/num_trials:.1f}秒")
        
        # 保存结果
        suffix = f'_first{max_trials}' if max_trials is not None else ''
        output_csv = f'exact_mi_results_{self.subject_id}{suffix}.csv'
        
        data_rows = []
        for result in results:
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
        df.to_csv(output_csv, index=False, encoding='utf-8-sig')
        
        if self.verbose:
            print(f"结果已保存至: {output_csv}")
        
        return results, df

def main():
    """主函数"""
    # 解析命令行参数
    max_trials = 3  # 默认只分析3个trials
    
    for arg in sys.argv:
        if arg.startswith('--trials='):
            try:
                max_trials = int(arg.split('=')[1])
            except ValueError:
                print("⚠️ 无效的trials参数")
    
    if '--help' in sys.argv:
        print("精确互信息分析 - 使用真实的信息论互信息")
        print("用法: python exact_mutual_information.py [--trials=N]")
        print("参数:")
        print("  --trials=N  : 分析的trials数量（默认3，建议≤10）")
        print("  --help      : 显示帮助")
        print("\n⚠️ 警告：每个trial需要约60秒，请谨慎设置trials数量")
        return
    
    # 时间估算警告
    estimated_time = max_trials * 60  # 每个trial约60秒
    print(f"⚠️ 时间警告：预计需要 {estimated_time/60:.1f} 分钟完成 {max_trials} 个trials")
    print("如果时间太长，请使用 --trials=N 减少trials数量")
    
    response = input("\n继续吗？ (y/N): ")
    if response.lower() != 'y':
        print("已取消")
        return
    
    # 创建精确分析器
    analyzer = ExactMutualInformationAnalyzer(
        subject_id='aw',
        random_state=42,
        verbose=True,
        n_jobs=4,  # 并行进程数
        mi_neighbors=3
    )
    
    # 运行分析
    start_time = time.time()
    results, df = analyzer.run_exact_analysis(max_trials=max_trials)
    total_time = time.time() - start_time
    
    print(f"\n🎉 精确分析完成!")
    print(f"总耗时: {total_time/60:.1f}分钟")
    print(f"每trial平均: {total_time/len(results):.1f}秒")

if __name__ == '__main__':
    main()