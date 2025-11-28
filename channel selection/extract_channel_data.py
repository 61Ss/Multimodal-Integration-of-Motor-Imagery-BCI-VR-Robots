import pandas as pd
import numpy as np
import os
from scipy.io import loadmat, savemat
from analyze_channels import ChannelAnalyzer

class ChannelDataExtractor:
    def __init__(self, results_csv_path, mat_file_path, output_mat_path):
        self.results_csv_path = results_csv_path
        self.mat_file_path = mat_file_path
        self.output_mat_path = output_mat_path
        self.target_channels = None
        
    def load_target_channels(self):
        """从分析结果CSV中加载目标通道"""
        # 读取分析结果，尝试不同的编码格式
        try:
            # 首先尝试UTF-8编码
            df = pd.read_csv(self.results_csv_path, encoding='utf-8')
        except UnicodeDecodeError:
            try:
                # 尝试GBK编码（Windows常用）
                df = pd.read_csv(self.results_csv_path, encoding='gbk')
            except UnicodeDecodeError:
                try:
                    # 尝试Latin-1编码
                    df = pd.read_csv(self.results_csv_path, encoding='latin1')
                except Exception as e:
                    raise Exception(f"无法读取CSV文件: {str(e)}")
        
        # 筛选出类别1和类别2的高低密度交集的交集通道
        target_df = df[df['category'] == '类别1和类别2的高低密度交集的交集']
        self.target_channels = target_df['channel'].unique().astype(int)
        
        print(f"成功加载目标通道 {len(self.target_channels)} 个: {self.target_channels}")
        return self
        
    def extract_and_save_data(self):
        """从MAT文件中提取目标通道数据并保存"""
        if self.target_channels is None:
            raise ValueError("请先调用load_target_channels方法加载目标通道")
        
        # 检查MAT文件是否存在
        if not os.path.exists(self.mat_file_path):
            raise FileNotFoundError(f"MAT文件不存在: {self.mat_file_path}")
        
        # 加载MAT文件数据（参照visualize_eeg_psd.py的加载逻辑）
        print(f"正在加载MAT文件: {self.mat_file_path}")
        mat_data = loadmat(self.mat_file_path)
        
        # 提取关键数据
        trials = mat_data['trials']  # 形状: (num_trials, time_points, channels)
        fs = mat_data['fs'][0, 0]  # 采样率
        info = mat_data['info'][0, 0]  # 信息结构体
        mrk = mat_data['mrk'][0, 0]  # 事件标记数据
        
        # 提取通道标签
        ch_names = [str(ch[0]) for ch in info['clab'][0]]
        
        # 确保通道索引有效
        num_channels = trials.shape[2]
        valid_channels = [ch for ch in self.target_channels if 0 <= ch < num_channels]
        invalid_channels = [ch for ch in self.target_channels if not (0 <= ch < num_channels)]
        
        if invalid_channels:
            print(f"警告: 以下通道超出范围将被忽略: {invalid_channels}")
        
        if not valid_channels:
            raise ValueError("没有有效的通道可供提取")
        
        # 提取目标通道数据
        extracted_trials = trials[:, :, valid_channels]
        
        # 创建要保存的数据结构
        save_data = {
            'trials': extracted_trials,
            'fs': fs,
            'target_channels': np.array(valid_channels),
            'info': info,
            'mrk': mrk
        }
        
        # 保存到新的MAT文件
        savemat(self.output_mat_path, save_data)
        print(f"成功保存提取的通道数据到: {self.output_mat_path}")
        print(f"提取的通道数量: {len(valid_channels)}")
        print(f"数据形状: {extracted_trials.shape}")

# 主函数
if __name__ == '__main__':
    # 配置文件路径
    results_csv_path = 'e:\\surf_selection\\channel_analysis_results.csv'
    mat_file_path = 'e:\\surf_selection\\100cc\\data_set_IVa_aw.mat'
    output_mat_path = 'e:\\surf_selection\\data_set_IVa_aw_extracted.mat'
    
    # 创建提取器实例
    extractor = ChannelDataExtractor(
        results_csv_path=results_csv_path,
        mat_file_path=mat_file_path,
        output_mat_path=output_mat_path
    )
    
    # 加载目标通道
    extractor.load_target_channels()
    
    # 提取并保存数据
    extractor.extract_and_save_data()