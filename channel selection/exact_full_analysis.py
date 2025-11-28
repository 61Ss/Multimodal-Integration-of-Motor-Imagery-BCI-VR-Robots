#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
精确互信息分析 - 全量280个trials版本

🎯 特性：
- 使用真实的信息论互信息计算
- 支持进度保存和恢复
- 优化的并行处理
- 详细的时间估算和进度跟踪
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
import pickle
import json
from functools import partial
from multiprocessing import Pool, cpu_count
from datetime import datetime, timedelta

# 解决Windows上KMeans的内存泄漏警告
os.environ['OMP_NUM_THREADS'] = '2'

# 导入现有的数据加载函数
from visualize_eeg_psd import load_eeg_data

class FullExactMutualInformationAnalyzer:
    """
    全量精确EEG互信息分析器
    
    支持：
    1. 280个trials的完整分析
    2. 进度保存和恢复
    3. 智能并行处理
    4. 详细的进度跟踪
    """
    
    def __init__(self, subject_id='aw', random_state=42, verbose=True, 
                 n_jobs=None, mi_neighbors=3, checkpoint_dir='checkpoints'):
        """
        初始化全量精确分析器
        
        Parameters:
        -----------
        n_jobs : int or None
            并行计算的进程数，None表示使用所有CPU核心
        mi_neighbors : int
            互信息计算中的邻居数（影响精度和速度）
        checkpoint_dir : str
            检查点保存目录
        """
        self.subject_id = subject_id
        self.random_state = random_state
        self.verbose = verbose
        self.n_jobs = n_jobs if n_jobs is not None else min(cpu_count(), 8)  # 限制最大进程数
        self.mi_neighbors = mi_neighbors
        self.checkpoint_dir = checkpoint_dir
        
        # 创建检查点目录
        os.makedirs(checkpoint_dir, exist_ok=True)
        
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
        
        # 时间跟踪
        self.start_time = None
        self.trial_times = []
        
    def save_checkpoint(self, results, trial_idx, metadata=None):
        """保存检查点"""
        checkpoint_data = {
            'results': results,
            'completed_trials': trial_idx + 1,
            'total_trials': len(self.trials),
            'timestamp': datetime.now().isoformat(),
            'subject_id': self.subject_id,
            'metadata': metadata or {}
        }
        
        checkpoint_file = os.path.join(
            self.checkpoint_dir, 
            f'checkpoint_{self.subject_id}_trial_{trial_idx+1}.pkl'
        )
        
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(checkpoint_data, f)
    
    def load_checkpoint(self):
        """加载最新的检查点"""
        checkpoint_files = []
        for file in os.listdir(self.checkpoint_dir):
            if file.startswith(f'checkpoint_{self.subject_id}_') and file.endswith('.pkl'):
                trial_num = int(file.split('_trial_')[1].split('.pkl')[0])
                checkpoint_files.append((trial_num, file))
        
        if not checkpoint_files:
            return None, 0
        
        # 找到最新的检查点
        latest_trial, latest_file = max(checkpoint_files)
        checkpoint_path = os.path.join(self.checkpoint_dir, latest_file)
        
        try:
            with open(checkpoint_path, 'rb') as f:
                checkpoint_data = pickle.load(f)
            
            if self.verbose:
                print(f"📁 恢复检查点: {checkpoint_data['completed_trials']}/{checkpoint_data['total_trials']} trials")
            
            return checkpoint_data, latest_trial
        
        except Exception as e:
            if self.verbose:
                print(f"⚠️ 加载检查点失败: {e}")
            return None, 0
    
    def estimate_remaining_time(self, completed_trials, total_trials):
        """估算剩余时间"""
        if len(self.trial_times) < 2:
            return "计算中..."
        
        avg_time_per_trial = np.mean(self.trial_times)
        remaining_trials = total_trials - completed_trials
        remaining_seconds = avg_time_per_trial * remaining_trials
        
        if remaining_seconds < 60:
            return f"{remaining_seconds:.0f}秒"
        elif remaining_seconds < 3600:
            return f"{remaining_seconds/60:.1f}分钟"
        else:
            hours = remaining_seconds // 3600
            minutes = (remaining_seconds % 3600) // 60
            return f"{hours:.0f}小时{minutes:.0f}分钟"
    
    def calculate_exact_mutual_information_matrix(self, data, band_name=""):
        """计算精确的互信息矩阵"""
        num_channels = data.shape[1]
        mi_matrix = np.zeros((num_channels, num_channels))
        
        # 计算所有通道对的互信息
        channel_pairs = [(i, j) for i in range(num_channels) for j in range(i + 1, num_channels)]
        total_pairs = len(channel_pairs)
        
        start_time = time.time()
        
        # 并行计算
        mi_func = partial(self._compute_exact_mi_pair, data=data)
        with Pool(self.n_jobs) as pool:
            mi_values = pool.map(mi_func, channel_pairs)
        
        calc_time = time.time() - start_time
        
        # 填充对称矩阵
        for (i, j), mi_value in zip(channel_pairs, mi_values):
            mi_matrix[i, j] = mi_value
            mi_matrix[j, i] = mi_value
        
        # 对角线设为0
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
        trial_start_time = time.time()
        
        trial_data = self.trials[trial_idx]
        
        # 存储各频段的互信息矩阵
        band_mi_matrices = {}
        
        for band_name, band_range in self.bands.items():
            # 滤波
            filtered_data = self.apply_bandpass_filter(trial_data, band_range, self.fs)
            
            # 计算精确互信息矩阵
            mi_matrix = self.calculate_exact_mutual_information_matrix(filtered_data, band_name)
            band_mi_matrices[band_name] = mi_matrix
        
        trial_time = time.time() - trial_start_time
        self.trial_times.append(trial_time)
        
        return band_mi_matrices
    
    def load_and_preprocess_data(self):
        """加载并预处理EEG数据"""
        if self.verbose:
            print("🔄 加载数据...")
        
        trials, fs, ch_names, ch_pos, trial_labels, regions = load_eeg_data(self.subject_id)
        
        self.trials = trials
        self.fs = fs
        self.ch_names = ch_names
        self.ch_pos = ch_pos
        self.trial_labels = trial_labels
        self.regions = regions
        
        if self.verbose:
            print(f"✓ 数据: {trials.shape}, 进程: {self.n_jobs}")
        
        return trials
    
    def calculate_density_labels_fast(self):
        """快速计算密度标签"""
        if self.verbose:
            print("🔄 计算密度标签...")
        
        num_trials, num_timepoints, num_channels = self.trials.shape
        
        # 大幅采样以加速PCA
        sample_step = max(1, num_timepoints // 50)
        sampled_trials = self.trials[:, ::sample_step, :]
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
            print(f"✓ 密度标签: High={high_count}, Low={low_count}")
        
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
    
    def run_full_exact_analysis(self, resume=True):
        """运行完整的精确分析"""
        if self.verbose:
            print("🎯 全量精确EEG互信息分析 - 280个trials")
        
        self.start_time = time.time()
        
        # 加载数据
        self.load_and_preprocess_data()
        
        # 计算密度标签
        self.calculate_density_labels_fast()
        
        num_trials = self.trials.shape[0]
        results = []
        start_trial = 0
        
        # 尝试恢复检查点
        if resume:
            checkpoint_data, last_completed = self.load_checkpoint()
            if checkpoint_data:
                results = checkpoint_data['results']
                start_trial = checkpoint_data['completed_trials']
                
                # 恢复信息已在load_checkpoint中显示
                
                if start_trial >= num_trials:
                    if self.verbose:
                        print("✅ 所有trials已完成!")
                    return results, None
        
        if self.verbose:
            estimated_total_time = (num_trials - start_trial) * 60  # 假设每trial 60秒
            print(f"🚀 开始分析: {start_trial}->{num_trials-1} (预计{estimated_total_time/3600:.1f}h)")
        
        # 分析剩余的trials
        for trial_idx in range(start_trial, num_trials):
            trial_overall_start = time.time()
            
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
            
            trial_total_time = time.time() - trial_overall_start
            
            # 进度显示 - 只在关键节点显示
            completed = trial_idx - start_trial + 1
            total_remaining = num_trials - start_trial
            progress_pct = 100 * completed / total_remaining
            
            # 每10个trials保存一次检查点并显示进度
            if (trial_idx + 1) % 10 == 0:
                self.save_checkpoint(results, trial_idx)
                if self.verbose and len(self.trial_times) > 0:
                    avg_time = np.mean(self.trial_times)
                    remaining_time_str = self.estimate_remaining_time(trial_idx + 1, num_trials)
                    print(f"✓ {trial_idx+1}/{num_trials} ({progress_pct:.0f}%) - {avg_time:.0f}s/trial - 剩余{remaining_time_str}")
            elif (trial_idx + 1) % 5 == 0:
                self.save_checkpoint(results, trial_idx)
        
        # 最终检查点
        if results:
            self.save_checkpoint(results, num_trials - 1)
        
        total_time = time.time() - self.start_time
        
        if self.verbose:
            print(f"\n🎉 分析完成! 用时{total_time/3600:.1f}h, 平均{total_time/num_trials:.0f}s/trial")
        
        # 保存最终结果
        output_csv = f'exact_full_mi_results_{self.subject_id}.csv'
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
            print(f"✓ 结果保存: {output_csv}")
        
        return results, df

def main():
    """主函数"""
    if '--help' in sys.argv:
        print("全量精确互信息分析 - 280个trials")
        print("用法: python exact_full_analysis.py [选项]")
        print("选项:")
        print("  --no-resume    : 不恢复检查点，重新开始")
        print("  --jobs=N       : 设置并行进程数")
        print("  --help         : 显示帮助")
        print("\n⚠️ 警告：预计需要4-6小时完成全部280个trials")
        return
    
    # 解析参数
    resume = '--no-resume' not in sys.argv
    n_jobs = None
    
    for arg in sys.argv:
        if arg.startswith('--jobs='):
            try:
                n_jobs = int(arg.split('=')[1])
            except ValueError:
                print("⚠️ 无效的jobs参数")
    
    # 时间警告
    print("🚀 全量精确EEG互信息分析")
    print("⚠️ 预计耗时4-6小时，使用真实互信息计算")
    resume_msg = "恢复进度" if resume else "重新开始"
    print(f"📋 {resume_msg}, 进程数: {n_jobs or '自动'}, 支持中断恢复")
    
    response = input("确认开始全量分析吗？ (y/N): ")
    if response.lower() != 'y':
        print("已取消")
        return
    
    # 创建分析器
    analyzer = FullExactMutualInformationAnalyzer(
        subject_id='aw',
        random_state=42,
        verbose=True,
        n_jobs=n_jobs,
        mi_neighbors=3
    )
    
    # 运行分析
    try:
        results, df = analyzer.run_full_exact_analysis(resume=resume)
        print(f"\n🎉 全量分析成功完成! 共处理 {len(results)} 个trials")
    except KeyboardInterrupt:
        print(f"\n⏸️ 分析被用户中断")
        print("进度已保存，可以稍后使用相同命令恢复")
    except Exception as e:
        print(f"\n❌ 分析过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()