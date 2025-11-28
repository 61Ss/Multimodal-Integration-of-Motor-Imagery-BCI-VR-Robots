import csv

# 文件路径配置
INPUT_FILE = r'C:\Users\thzx4\Desktop\BCI\TryData\A01session1Run1.csv'
FILTERED_FILE = r'C:\Users\thzx4\Desktop\BCI\TryData\A01session1Run1_filtered_data.csv'
LABEL_FILE = r'C:\Users\thzx4\Desktop\BCI\TryData\A01session1Run1_label.csv'
MERGED_FILE = r'C:\Users\thzx4\Desktop\BCI\TryData\A01session1Run1_merged_data.csv'
PROCESSED_FILE = r'C:\Users\thzx4\Desktop\BCI\TryData\A01session1Run1_processed_trials.csv'

# 配置参数
SAMPLING_RATE = 128  # 采样率: 128样本/秒
EXCLUDE_SECONDS = 40  # 排除前30秒
KEEP_SECONDS = 400  # 保留之后360秒
SAMPLES_PER_LABEL = 128 * 10  # 每个标签对应的数据行数(128Hz * 9秒)
NUM_LABELS = 40  # 预期标签总数
TRIAL_DURATION = 10  # 原始trial时长(秒)
EXCLUDE_FIRST = 3  # 排除前N秒
EXCLUDE_LAST = 3  # 排除后N秒

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


# 数据处理步骤 1: 筛选数据
def filter_data():
    exclude_rows = EXCLUDE_SECONDS * SAMPLING_RATE
    keep_rows = KEEP_SECONDS * SAMPLING_RATE

    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as infile, \
                open(FILTERED_FILE, 'w', encoding='utf-8', newline='') as outfile:

            reader = csv.reader(infile)
            writer = csv.writer(outfile)

            # 读取并写入元数据行(第一行)和标题行(第二行)
            metadata_row = next(reader)
            header_row = next(reader)
            writer.writerow(metadata_row)
            writer.writerow(header_row)

            # 跳过需要排除的行
            for _ in range(exclude_rows):
                next(reader)

            # 写入需要保留的行
            for _ in range(keep_rows):
                try:
                    row = next(reader)
                    writer.writerow(row)
                except StopIteration:
                    print(f"警告: 文件数据不足，实际只提取了{_}行数据")
                    break

        print(f"数据筛选完成！输出文件: {FILTERED_FILE}")

    except FileNotFoundError:
        print(f"错误: 输入文件 '{INPUT_FILE}' 未找到")
    except Exception as e:
        print(f"处理文件时发生错误: {str(e)}")


# 数据处理步骤 2: 合并标签与数据
def merge_data_with_labels():
    labels = []
    try:
        with open(LABEL_FILE, 'r', encoding='utf-8') as labelfile:
            label_reader = csv.DictReader(labelfile)
            for row in label_reader:
                trial_type = row['trial_type']
                if trial_type in LABEL_MAPPING:
                    mapped_label = LABEL_MAPPING[trial_type]
                    labels.append(mapped_label)
                else:
                    print(f"警告: 未知标签'{trial_type}'，已跳过")

                if len(labels) >= NUM_LABELS:
                    break  # 确保只读取前40个标签

        if len(labels) != NUM_LABELS:
            print(f"警告: 标签数量不匹配，预期{NUM_LABELS}个，实际{len(labels)}个")

        with open(FILTERED_FILE, 'r', encoding='utf-8') as datafile, \
                open(MERGED_FILE, 'w', encoding='utf-8', newline='') as outfile:

            datafile.readline()
            data_reader = csv.DictReader(datafile)

            original_headers = data_reader.fieldnames
            selected_columns = ['Timestamp']  # 保留时间戳
            for channel in CHANNELS_TO_KEEP:
                if channel in original_headers:
                    selected_columns.append(channel)

            selected_columns.append('trial_type')

            writer = csv.DictWriter(outfile, fieldnames=selected_columns)
            writer.writeheader()

            for label_idx, label in enumerate(labels):
                print(f"正在处理标签 {label_idx + 1}/{len(labels)}: {label}")

                for row_idx in range(SAMPLES_PER_LABEL):
                    try:
                        row = next(data_reader)
                        new_row = {col: row[col] for col in selected_columns if col != 'trial_type'}
                        new_row['trial_type'] = label
                        writer.writerow(new_row)
                    except StopIteration:
                        print(f"警告: 数据不足，在处理标签{label}时提前结束")
                        return

        print(f"数据合并完成！输出文件: {MERGED_FILE}")

    except FileNotFoundError as e:
        print(f"错误: 文件未找到 - {str(e)}")
    except Exception as e:
        print(f"处理文件时发生错误: {str(e)}")


# 数据处理步骤 3: 处理试次
def process_trials():
    SAMPLES_PER_SECOND = 128
    TOTAL_SAMPLES_PER_TRIAL = TRIAL_DURATION * SAMPLES_PER_SECOND
    SAMPLES_TO_EXCLUDE_FIRST = EXCLUDE_FIRST * SAMPLES_PER_SECOND
    SAMPLES_TO_EXCLUDE_LAST = EXCLUDE_LAST * SAMPLES_PER_SECOND
    SAMPLES_TO_KEEP = TOTAL_SAMPLES_PER_TRIAL - SAMPLES_TO_EXCLUDE_FIRST - SAMPLES_TO_EXCLUDE_LAST

    try:
        with open(MERGED_FILE, 'r', encoding='utf-8') as infile, \
                open(PROCESSED_FILE, 'w', encoding='utf-8', newline='') as outfile:

            reader = csv.DictReader(infile)
            writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames)
            writer.writeheader()

            trial_counter = 0
            row_in_trial = 0

            for row in reader:
                if row_in_trial >= TOTAL_SAMPLES_PER_TRIAL:
                    trial_counter += 1
                    row_in_trial = 0

                if SAMPLES_TO_EXCLUDE_FIRST <= row_in_trial < (TOTAL_SAMPLES_PER_TRIAL - SAMPLES_TO_EXCLUDE_LAST):
                    writer.writerow(row)

                row_in_trial += 1

            print(f"数据处理完成！输出文件: {PROCESSED_FILE}")

    except FileNotFoundError:
        print(f"错误: 输入文件 '{MERGED_FILE}' 未找到")
    except Exception as e:
        print(f"处理文件时发生错误: {str(e)}")


# 运行所有步骤
def main():
    filter_data()
    merge_data_with_labels()
    process_trials()


if __name__ == "__main__":
    main()
