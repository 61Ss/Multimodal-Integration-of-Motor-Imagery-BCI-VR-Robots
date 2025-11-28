import csv

# 定义文件路径
input_file = r'e:\self_data_surf\data\A01\A01session1Run2_data.csv'
output_file = r'e:\self_data_surf\data\A01\A01session1Run2_filtered_data.csv'

# 配置参数
SAMPLING_RATE = 128  # 采样率: 128样本/秒
EXCLUDE_SECONDS = 30  # 排除前30秒
KEEP_SECONDS = 360    # 保留之后360秒

# 计算需要排除和保留的行数
exclude_rows = EXCLUDE_SECONDS * SAMPLING_RATE
keep_rows = KEEP_SECONDS * SAMPLING_RATE

try:
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8', newline='') as outfile:

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

    print(f"数据提取完成！\n排除: {EXCLUDE_SECONDS}秒 ({exclude_rows}行)\n保留: {KEEP_SECONDS}秒 ({keep_rows}行)\n输出文件: {output_file}")

except FileNotFoundError:
    print(f"错误: 输入文件 '{input_file}' 未找到")
except Exception as e:
    print(f"处理文件时发生错误: {str(e)}")