import pandas as pd
import numpy as np

class ChannelAnalyzer:
    def __init__(self, file_path, k=10):
        self.file_path = file_path
        self.k = k  # 前k个通道
        self.df = None
        # 定义通道列名
        self.channel_columns = [f'channel_{i}' for i in range(1, self.k+1)]  # channel_1到channel_k
        
    def load_data(self):
        """加载CSV数据"""
        # 读取CSV时指定编码格式为UTF-8
        self.df = pd.read_csv(self.file_path, encoding='utf-8')
        print(f"成功加载数据，共{len(self.df)}行")
        return self
        
    def get_channel_union(self, original_label=None, density_label=None):
        """获取指定条件下所有trial的通道并集"""
        # 筛选数据
        mask = pd.Series([True]*len(self.df))
        
        if original_label is not None:
            mask &= (self.df['original_label'] == original_label)
        if density_label is not None:
            mask &= (self.df['density_label'] == density_label)
        
        filtered_df = self.df[mask]
        print(f"筛选出{len(filtered_df)}行数据")
        
        # 收集所有通道并集
        all_channels = set()
        for _, row in filtered_df.iterrows():
            channels = [row[col] for col in self.channel_columns if pd.notna(row[col])]
            all_channels.update(channels)
        
        return list(all_channels)
        
    def analyze_all_categories(self):
        """分析所有四个类别，并计算交集"""
        categories = [
            (1, 'low', '类别1，low_density'),
            (2, 'low', '类别2，low_density'),
            (1, 'high', '类别1，high_density'),
            (2, 'high', '类别2，high_density')
        ]
        
        results = {}
        # 先获取所有类别的通道并集
        for original_label, density_label, category_name in categories:
            print(f"\n分析{category_name}...")
            channels = self.get_channel_union(
                original_label=original_label,
                density_label=density_label
            )
            results[category_name] = channels
            print(f"{category_name}通道并集 ({len(channels)}个): {channels}")
        
        # 计算类别1的low_density和high_density的交集
        cat1_low = set(results['类别1，low_density'])
        cat1_high = set(results['类别1，high_density'])
        cat1_intersection = list(cat1_low & cat1_high)
        results['类别1，low_density与high_density的交集'] = cat1_intersection
        print(f"\n类别1，low_density与high_density的交集 ({len(cat1_intersection)}个): {cat1_intersection}")
        
        # 计算类别2的low_density和high_density的交集
        cat2_low = set(results['类别2，low_density'])
        cat2_high = set(results['类别2，high_density'])
        cat2_intersection = list(cat2_low & cat2_high)
        results['类别2，low_density与high_density的交集'] = cat2_intersection
        print(f"类别2，low_density与high_density的交集 ({len(cat2_intersection)}个): {cat2_intersection}")
        
        # 计算两个交集之间的交集
        cat1_cat2_intersection = list(set(cat1_intersection) & set(cat2_intersection))
        results['类别1和类别2的高低密度交集的交集'] = cat1_cat2_intersection
        print(f"类别1和类别2的高低密度交集的交集 ({len(cat1_cat2_intersection)}个): {cat1_cat2_intersection}")
        
        return results
        
    def save_results(self, results, output_file):
        """只保存类别1和类别2的高低密度交集的交集到CSV文件"""
        target_category = '类别1和类别2的高低密度交集的交集'
        # 写入文件时指定编码格式为UTF-8，并设置newline参数确保跨平台一致性
        with open(output_file, 'w', encoding='utf-8', newline='\n') as f:
            f.write('category,channel\n')
            if target_category in results:
                for channel in results[target_category]:
                    f.write(f'{target_category},{channel}\n')
        print(f"结果已保存到{output_file}，仅包含{target_category}，编码格式为UTF-8")

# 主函数
if __name__ == '__main__':
    # 创建分析器实例
    analyzer = ChannelAnalyzer(
        file_path='e:\\surf_selection\\exact_full_mi_results_aw.csv',
        k=10  # 提取前10个通道
    )
    
    # 加载数据
    analyzer.load_data()
    
    # 分析所有类别
    results = analyzer.analyze_all_categories()
    
    # 保存结果
    analyzer.save_results(results, 'e:\\surf_selection\\channel_analysis_results.csv')