import csv

# 文件路径配置
DATA_FILE = r'e:\self_data_surf\data\A01\A01session1Run2_filtered_data.csv'
LABEL_FILE = r'e:\self_data_surf\data\A01\A01session1Run2_label.csv'
OUTPUT_FILE = r'e:\self_data_surf\data\A01\A01session1Run2_merged_data.csv'

# 配置参数
SAMPLES_PER_LABEL = 128 * 9  # 每个标签对应的数据行数(128Hz * 9秒)
NUM_LABELS = 40              # 预期标签总数

# 标签映射关系
LABEL_MAPPING = {
    'left': 0,
    'right': 1
}

# 需要保留的32个EEG通道
CHANNELS_TO_KEEP = [
    'EEG.Cz', 'EEG.FCz', 'EEG.Fz', 'EEG.AF3', 'EEG.F7', 'EEG.F3', 'EEG.F1', 'EEG.FC1',
    'EEG.C1', 'EEG.C3', 'EEG.FC3', 'EEG.FC5', 'EEG.T7', 'EEG.CP3', 'EEG.CP1', 'EEG.P7',
    'EEG.O1', 'EEG.CPz', 'EEG.O2', 'EEG.P8', 'EEG.CP2', 'EEG.CP4', 'EEG.T8', 'EEG.FC6',
    'EEG.FC4', 'EEG.C4', 'EEG.C2', 'EEG.FC2', 'EEG.F2', 'EEG.F4', 'EEG.F8', 'EEG.AF4'
]

def main():
    try:
        # 读取标签数据
        labels = []
        with open(LABEL_FILE, 'r', encoding='utf-8') as labelfile:
            label_reader = csv.DictReader(labelfile)
            for row in label_reader:
                trial_type = row['trial_type']
                # 应用标签映射
                if trial_type in LABEL_MAPPING:
                    mapped_label = LABEL_MAPPING[trial_type]
                    labels.append(mapped_label)
                else:
                    print(f"警告: 未知标签'{trial_type}'，已跳过")
                
                if len(labels) >= NUM_LABELS:
                    break  # 确保只读取前40个标签
        
        # 验证标签数量
        if len(labels) != NUM_LABELS:
            print(f"警告: 标签数量不匹配，预期{NUM_LABELS}个，实际{len(labels)}个")
        
        # 读取数据文件并处理
        with open(DATA_FILE, 'r', encoding='utf-8') as datafile, \
             open(OUTPUT_FILE, 'w', encoding='utf-8', newline='') as outfile:
            
            # 跳过第一行元数据
            datafile.readline()
            # 从第二行开始读取表头
            data_reader = csv.DictReader(datafile)
            
            # 获取原始数据的表头
            original_headers = data_reader.fieldnames
            
            # 找到需要保留的通道对应的列
            selected_columns = ['Timestamp']  # 保留时间戳
            for channel in CHANNELS_TO_KEEP:
                if channel in original_headers:
                    selected_columns.append(channel)
                else:
                    print(f"警告: 通道'{channel}'在数据文件中不存在，已跳过")
            
            # 添加标签列
            selected_columns.append('trial_type')
            
            # 创建写入器
            writer = csv.DictWriter(outfile, fieldnames=selected_columns)
            writer.writeheader()
            
            # 处理每个标签对应的数据块
            for label_idx, label in enumerate(labels):
                print(f"正在处理标签 {label_idx+1}/{len(labels)}: {label}")
                
                # 读取当前标签对应的数据行
                for row_idx in range(SAMPLES_PER_LABEL):
                    try:
                        row = next(data_reader)
                        # 只保留选中的列和添加标签
                        new_row = {col: row[col] for col in selected_columns if col != 'trial_type'}
                        new_row['trial_type'] = label  # 使用映射后的标签
                        writer.writerow(new_row)
                    except StopIteration:
                        print(f"警告: 数据不足，在处理标签{label}时提前结束")
                        return
        
        print(f"数据合并完成！输出文件: {OUTPUT_FILE}")
    
    except FileNotFoundError as e:
        print(f"错误: 文件未找到 - {str(e)}")
    except Exception as e:
        print(f"处理文件时发生错误: {str(e)}")

if __name__ == "__main__":
    main()