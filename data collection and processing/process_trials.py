import csv

# 文件路径配置
INPUT_FILE = r'e:\self_data_surf\data\A01\A01session1Run2_merged_data.csv'
OUTPUT_FILE = r'e:\self_data_surf\data\A01\A01session1Run2_processed_trials.csv'

# 配置参数
SAMPLES_PER_SECOND = 128  # 采样率
TRIAL_DURATION = 9        # 原始trial时长(秒)
EXCLUDE_FIRST = 2         # 排除前N秒
EXCLUDE_LAST = 3          # 排除后N秒

# 计算每个trial的行数
TOTAL_SAMPLES_PER_TRIAL = TRIAL_DURATION * SAMPLES_PER_SECOND
SAMPLES_TO_EXCLUDE_FIRST = EXCLUDE_FIRST * SAMPLES_PER_SECOND
SAMPLES_TO_EXCLUDE_LAST = EXCLUDE_LAST * SAMPLES_PER_SECOND
SAMPLES_TO_KEEP = TOTAL_SAMPLES_PER_TRIAL - SAMPLES_TO_EXCLUDE_FIRST - SAMPLES_TO_EXCLUDE_LAST

def main():
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as infile, \
             open(OUTPUT_FILE, 'w', encoding='utf-8', newline='') as outfile:
            
            reader = csv.DictReader(infile)
            writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames)
            writer.writeheader()
            
            trial_counter = 0
            row_in_trial = 0
            
            for row in reader:
                # 计算当前行在trial中的位置
                if row_in_trial >= TOTAL_SAMPLES_PER_TRIAL:
                    # 开始新的trial
                    trial_counter += 1
                    row_in_trial = 0
                
                # 只保留中间部分的数据
                if SAMPLES_TO_EXCLUDE_FIRST <= row_in_trial < (TOTAL_SAMPLES_PER_TRIAL - SAMPLES_TO_EXCLUDE_LAST):
                    writer.writerow(row)
                
                row_in_trial += 1
            
            print(f"数据处理完成！\n总处理trials: {trial_counter + 1}\n每个trial排除前{EXCLUDE_FIRST}秒({SAMPLES_TO_EXCLUDE_FIRST}行)和后{EXCLUDE_LAST}秒({SAMPLES_TO_EXCLUDE_LAST}行)\n每个trial保留中间{(TRIAL_DURATION - EXCLUDE_FIRST - EXCLUDE_LAST)}秒({SAMPLES_TO_KEEP}行)\n输出文件: {OUTPUT_FILE}")
    
    except FileNotFoundError:
        print(f"错误: 输入文件 '{INPUT_FILE}' 未找到")
    except Exception as e:
        print(f"处理文件时发生错误: {str(e)}")

if __name__ == "__main__":
    main()