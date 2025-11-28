import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.signal import welch
import mne
from mne.channels import make_dig_montage
from mne.viz import plot_topomap

# ---------------------- 配置参数 ----------------------
# 基本设置
SUBJECT_ID = ['aa','al']  # 受试者ID，支持单个字符串或列表形式
TRIAL_INDEX = 0  # 要可视化的单个trial索引 (0-based)
INPUT_DIR = '100cc'  # 输入数据目录
OUTPUT_DIR = 'single_subject_psd_visualization'  # 输出图像目录
CLS = True  # 是否按标签分类可视化，默认为False
AVERAGE_ALL = True  # 是否计算所有受试者的平均值，仅当SUBJECT_ID为列表时生效

# 频段定义 (Hz)
BANDS = {
    'alpha': (7, 13),
    'beta': (14, 30),
    'alpha+beta': (7, 30)
}

# 坐标缩放因子 (与电极可视化脚本保持一致)
COORD_SCALE = 0.1

# 脑区映射配置
REGION_MAP = {
    'F': 'Frontal', 
    'A': 'Frontal', 
    'C': 'Central', 
    'P': 'Parietal', 
    'O': 'Occipital', 
    'I': 'Occipital', 
    'T': 'Temporal'
}

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------- 数据加载 ----------------------
def load_eeg_data(subject_id):
    """加载处理后的EEG数据，包括电极坐标和trial标签"""
    filename = f'data_set_IVa_{subject_id}.mat'
    file_path = os.path.join(INPUT_DIR, filename)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件 {file_path} 不存在")

    mat_data = loadmat(file_path)

    # 提取关键数据
    trials = mat_data['trials']  # 形状: (num_trials, time_points, channels)
    fs = mat_data['fs'][0, 0]  # 采样率
    info = mat_data['info'][0, 0]  # 信息结构体
    mrk = mat_data['mrk'][0, 0]  # 事件标记数据
    trial_labels = mrk['y'].flatten()  # trial标签

    # 提取通道标签和电极坐标
    ch_names = [str(ch[0]) for ch in info['clab'][0]]
    x_pos = info['xpos'].flatten() * COORD_SCALE  # 应用坐标缩放
    y_pos = info['ypos'].flatten() * COORD_SCALE

    # 确定每个电极的脑区
    regions = get_brain_regions(ch_names, REGION_MAP)

    # 创建电极位置字典 (MNE需要3D坐标，z设为0)
    ch_pos = {
        ch_names[i]: [x_pos[i], y_pos[i], 0.0]
        for i in range(len(ch_names))
    }

    return trials, fs, ch_names, ch_pos, trial_labels, regions


# ---------------------- 脑区分类 ----------------------
def get_brain_regions(ch_names, region_map):
    """根据电极名称前缀确定每个电极的脑区"""
    regions = []
    for ch in ch_names:
        # 取电极名称的第一个字符作为前缀
        prefix = ch[0].upper()
        region = region_map.get(prefix, 'Other')
        regions.append(region)
    return regions

# ---------------------- PSD计算 ----------------------
def calculate_band_power(data, fs, band):
    # "计算指定频段的功率谱密度(PSD)"
    fmin, fmax = band

    # 使用Welch方法计算PSD
    freqs, psd = welch(
        data,
        fs=fs,
        nperseg=min(128, data.shape[0]),  # 根据时间点数量调整窗口大小
        noverlap=64,
        axis=0
    )

    # 提取目标频段的PSD并积分
    freq_mask = (freqs >= fmin) & (freqs <= fmax)
    band_power = np.trapz(psd[freq_mask, :], freqs[freq_mask], axis=0)

    return band_power

# ---------------------- 可视化函数 ----------------------
def plot_single_trial_psd(single_trial_data, fs, ch_names, ch_pos, output_prefix):
    # "可视化单个trial的各通道PSD"
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'Single Trial {TRIAL_INDEX} - Band Power Distribution', fontsize=16)

    # 创建MNE信息对象和电极位置
    info = mne.create_info(ch_names=ch_names, sfreq=fs, ch_types='eeg')
    montage = make_dig_montage(ch_pos=ch_pos, coord_frame='head')
    info.set_montage(montage)

    for i, (band_name, band) in enumerate(BANDS.items()):
        # 计算该频段功率
        band_power = calculate_band_power(single_trial_data, fs, band)

        # 绘制拓扑图
        im, _ = plot_topomap(band_power, info, axes=axes[i], show=False,
                             cmap='viridis', vlim=(np.min(band_power), np.max(band_power)))
        axes[i].set_title(f'{band_name} ({band[0]}-{band[1]} Hz)')
        fig.colorbar(im, ax=axes[i], label='Power (μV²/Hz)')

    output_path = os.path.join(OUTPUT_DIR, f'{output_prefix}_single_trial_topomap.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"单trial拓扑图已保存至: {output_path}")


def plot_average_psd(all_trials_data, fs, ch_names, ch_pos, output_prefix):
    # "可视化所有trial平均后的PSD"
    # 计算所有trial的平均功率
    avg_band_powers = {}

    for band_name, band in BANDS.items():
        band_powers = []

        # 遍历所有trial计算功率
        for trial in all_trials_data:
            power = calculate_band_power(trial, fs, band)
            band_powers.append(power)

        # 计算平均功率
        avg_band_powers[band_name] = np.mean(band_powers, axis=0)

    # 创建MNE信息对象和电极位置
    info = mne.create_info(ch_names=ch_names, sfreq=fs, ch_types='eeg')
    montage = make_dig_montage(ch_pos=ch_pos, coord_frame='head')
    info.set_montage(montage)

    # 绘制平均功率拓扑图
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'Average Across All Trials - Band Power Distribution', fontsize=16)

    for i, (band_name, band) in enumerate(BANDS.items()):
        # 绘制拓扑图
        im, _ = plot_topomap(avg_band_powers[band_name], info, axes=axes[i], show=False,
                            cmap='viridis', vlim=(np.min(avg_band_powers[band_name]), np.max(avg_band_powers[band_name])))
        axes[i].set_title(f'{band_name} ({band[0]}-{band[1]} Hz)')
        fig.colorbar(im, ax=axes[i], label='Average Power (μV²/Hz)')

    output_path = os.path.join(OUTPUT_DIR, f'{output_prefix}_average_topomap.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"平均拓扑图已保存至: {output_path}")

    # 绘制通道功率条形图
    fig, axes = plt.subplots(3, 1, figsize=(15, 18))
    fig.suptitle(f'Average Band Power by Channel', fontsize=16)
    
    # 按脑区对通道进行排序
    sorted_indices = np.argsort(regions)
    sorted_ch_names = [ch_names[i] for i in sorted_indices]
    sorted_regions = [regions[i] for i in sorted_indices]
    sorted_powers = {band: power[sorted_indices] for band, power in avg_band_powers.items()}
    
    # 获取唯一脑区和对应的颜色
    unique_regions = list(np.unique(regions))
    colors = plt.cm.get_cmap('tab10', len(unique_regions))
    region_colors = {region: colors(i) for i, region in enumerate(unique_regions)}
    bar_colors = [region_colors[region] for region in sorted_regions]
    
    for i, (band_name, power) in enumerate(sorted_powers.items()):
        axes[i].bar(sorted_ch_names, power, color=bar_colors)
        axes[i].set_title(f'{band_name} Band Power')
        axes[i].set_xlabel('Channels')
        axes[i].set_ylabel('Power (μV²/Hz)')
        axes[i].tick_params(axis='x', rotation=90)

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # 为标题留出空间
    output_path = os.path.join(OUTPUT_DIR, f'{output_prefix}_channel_barplot.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"通道功率条形图已保存至: {output_path}")

# ---------------------- 可视化函数 ----------------------
def plot_labeled_average_psd(all_trials_data, trial_labels, fs, ch_names, ch_pos, output_prefix):
    """按标签分组可视化平均PSD"""
    unique_labels = np.unique(trial_labels)
    num_labels = len(unique_labels)

    # 为每个标签计算平均功率
    label_band_powers = {}
    for label in unique_labels:
        # 获取当前标签的所有trial
        label_trials = all_trials_data[trial_labels == label]
        label_band_powers[label] = {}

        # 计算每个频段的平均功率
        for band_name, band in BANDS.items():
            band_powers = []
            for trial in label_trials:
                power = calculate_band_power(trial, fs, band)
                band_powers.append(power)
            label_band_powers[label][band_name] = np.mean(band_powers, axis=0)

    # 创建MNE信息对象
    info = mne.create_info(ch_names=ch_names, sfreq=fs, ch_types='eeg')
    montage = make_dig_montage(ch_pos=ch_pos, coord_frame='head')
    info.set_montage(montage)

    # 绘制每个标签的拓扑图和条形图
    for label in unique_labels:
        # 绘制拓扑图
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f'Label {label} - Average Band Power Distribution', fontsize=16)

        for i, (band_name, band) in enumerate(BANDS.items()):
            im, _ = plot_topomap(label_band_powers[label][band_name], info, axes=axes[i], show=False,
                                cmap='viridis', vlim=(np.min(label_band_powers[label][band_name]), np.max(label_band_powers[label][band_name])))
            axes[i].set_title(f'{band_name} ({band[0]}-{band[1]} Hz)')
            fig.colorbar(im, ax=axes[i], label='Average Power (μV²/Hz)')

        output_path = os.path.join(OUTPUT_DIR, f'{output_prefix}_label{label}_average_topomap.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"标签 {label} 的平均拓扑图已保存至: {output_path}")

        # 绘制通道功率条形图
        fig, axes = plt.subplots(3, 1, figsize=(15, 18))
        fig.suptitle(f'Label {label} - Average Band Power by Channel', fontsize=16)

        for i, (band_name, power) in enumerate(label_band_powers[label].items()):
            axes[i].bar(ch_names, power)
            axes[i].set_title(f'{band_name} Band Power')
            axes[i].set_xlabel('Channels')
            axes[i].set_ylabel('Power (μV²/Hz)')
            axes[i].tick_params(axis='x', rotation=90)

        plt.tight_layout(rect=[0, 0, 1, 0.96])  # 为标题留出空间
        output_path = os.path.join(OUTPUT_DIR, f'{output_prefix}_label{label}_channel_barplot.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"标签 {label} 的通道功率条形图已保存至: {output_path}")


# ---------------------- 可视化函数 ----------------------
def plot_brain_region_comparison(band_powers, regions, output_prefix, labels=None):
    """按脑区比较不同频段的平均功率，支持按标签分类
    Args:
        band_powers: 功率数据字典，格式为{band_name: powers_array}
                     当labels不为None时，格式为{label: {band_name: powers_array}}
        regions: 电极脑区列表
        output_prefix: 输出文件前缀
        labels: 可选参数，标签列表或None
    """
    # 处理按标签分类的情况
    if labels is not None:
        unique_labels = np.unique(labels)
        for label in unique_labels:
            # 提取当前标签的功率数据
            label_band_powers = band_powers[label]
            
            # 整理数据：将每个脑区的所有通道功率收集起来
            region_data = {}
            for band, powers in label_band_powers.items():
                region_data[band] = {}
                for power, region in zip(powers, regions):
                    if region not in region_data[band]:
                        region_data[band][region] = []
                    region_data[band][region].append(power)
            
            # 计算每个脑区的平均功率
            region_avg = {}
            for band, data in region_data.items():
                region_avg[band] = {region: np.mean(powers) for region, powers in data.items()}
            
            # 绘制条形图比较不同脑区的功率
            regions_list = list(region_avg[list(BANDS.keys())[0]].keys())
            n_regions = len(regions_list)
            bar_width = 0.2
            index = np.arange(n_regions)
            
            fig, ax = plt.subplots(figsize=(12, 6))
            for i, (band, avg_powers) in enumerate(region_avg.items()):
                powers = [avg_powers[region] for region in regions_list]
                ax.bar(index + i * bar_width, powers, bar_width, label=band)
            
            ax.set_xlabel('Brain Regions')
            ax.set_ylabel('Average Power (μV²/Hz)')
            ax.set_title(f'Label {label} - Average Band Power by Brain Region')
            ax.set_xticks(index + bar_width * (len(BANDS) - 1) / 2)
            ax.set_xticklabels(regions_list)
            ax.legend()
            
            output_path = os.path.join(OUTPUT_DIR, f'{output_prefix}_label{label}_brain_region_comparison.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"标签 {label} 的脑区功率对比图已保存至: {output_path}")
    else:
        # 处理不分类的情况（原有逻辑）
        # 整理数据：将每个脑区的所有通道功率收集起来
        region_data = {}
        for band, powers in band_powers.items():
            region_data[band] = {}
            for power, region in zip(powers, regions):
                if region not in region_data[band]:
                    region_data[band][region] = []
                region_data[band][region].append(power)
        
        # 计算每个脑区的平均功率
        region_avg = {}
        for band, data in region_data.items():
            region_avg[band] = {region: np.mean(powers) for region, powers in data.items()}
        
        # 绘制条形图比较不同脑区的功率
        regions_list = list(region_avg[list(BANDS.keys())[0]].keys())
        n_regions = len(regions_list)
        bar_width = 0.2
        index = np.arange(n_regions)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        for i, (band, avg_powers) in enumerate(region_avg.items()):
            powers = [avg_powers[region] for region in regions_list]
            ax.bar(index + i * bar_width, powers, bar_width, label=band)
        
        ax.set_xlabel('Brain Regions')
        ax.set_ylabel('Average Power (μV²/Hz)')
        ax.set_title('Average Band Power by Brain Region')
        ax.set_xticks(index + bar_width * (len(BANDS) - 1) / 2)
        ax.set_xticklabels(regions_list)
        ax.legend()
        
        output_path = os.path.join(OUTPUT_DIR, f'{output_prefix}_brain_region_comparison.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"脑区功率对比图已保存至: {output_path}")


# ---------------------- 主程序 ----------------------
if __name__ == '__main__':
    print("===== 开始EEG频段功率可视化 =====")

    try:
        # 确保SUBJECT_ID是列表格式
        if not isinstance(SUBJECT_ID, list):
            subject_ids = [SUBJECT_ID]
        else:
            subject_ids = SUBJECT_ID
            
        # 存储所有受试者的数据
        all_subjects_data = []
        all_subjects_labels = []
        all_subjects_regions = None
        fs = None
        ch_names = None
        ch_pos = None
        
        # 遍历每个受试者
        for subj_id in subject_ids:
            print(f"正在处理受试者: {subj_id}")
            # 加载数据（包含电极坐标和trial标签）
            trials, fs, ch_names, ch_pos, trial_labels, regions = load_eeg_data(subj_id)
            num_trials, num_timepoints, num_channels = trials.shape
            print(f"成功加载数据: {num_trials}个trials, {num_timepoints}个时间点, {num_channels}个通道")
            
            # 存储当前受试者数据
            all_subjects_data.append(trials)
            all_subjects_labels.append(trial_labels)
            
            # 验证电极配置一致性
            if all_subjects_regions is None:
                all_subjects_regions = regions
            else:
                assert regions == all_subjects_regions, f"受试者{subj_id}的电极配置与第一个受试者不一致"
            
            # 提取单个trial数据并可视化
            single_trial = trials[TRIAL_INDEX, :, :]
            plot_single_trial_psd(single_trial, fs, ch_names, ch_pos, f'{subj_id}_trial{TRIAL_INDEX}')
            
            # 根据CLS参数决定可视化方式
            if CLS:
                plot_labeled_average_psd(trials, trial_labels, fs, ch_names, ch_pos, subj_id)
            else:
                plot_average_psd(trials, fs, ch_names, ch_pos, subj_id)
            
            # 计算当前受试者的脑区功率
            avg_band_powers = {}
            for band_name, band in BANDS.items():
                band_powers = []
                for trial in trials:
                    power = calculate_band_power(trial, fs, band)
                    band_powers.append(power)
                avg_band_powers[band_name] = np.mean(band_powers, axis=0)
            
            # 绘制当前受试者的脑区功率对比
            if CLS:
                # 按标签分类的脑区功率对比
                label_band_powers = {}
                unique_labels = np.unique(trial_labels)
                for label in unique_labels:
                    label_trials = trials[trial_labels == label]
                    label_band_powers[label] = {}
                    for band_name, band in BANDS.items():
                        band_powers = []
                        for trial in label_trials:
                            power = calculate_band_power(trial, fs, band)
                            band_powers.append(power)
                        label_band_powers[label][band_name] = np.mean(band_powers, axis=0)
                plot_brain_region_comparison(label_band_powers, regions, subj_id, trial_labels)
            else:
                plot_brain_region_comparison(avg_band_powers, regions, subj_id)
        
        # 如果需要计算所有受试者的平均值
        if AVERAGE_ALL and len(subject_ids) > 1:
            print("正在计算所有受试者的平均值...")
            # 合并所有受试者的数据 (subjects, trials, timepoints, channels)
            combined_trials = np.concatenate(all_subjects_data, axis=0)
            combined_labels = np.concatenate(all_subjects_labels, axis=0)
            output_prefix = 'all_subjects_average'
            
            # 可视化平均结果
            if CLS:
                plot_labeled_average_psd(combined_trials, combined_labels, fs, ch_names, ch_pos, output_prefix)
            else:
                plot_average_psd(combined_trials, fs, ch_names, ch_pos, output_prefix)
            
            # 计算平均脑区功率
            avg_band_powers = {}
            for band_name, band in BANDS.items():
                band_powers = []
                for trial in combined_trials:
                    power = calculate_band_power(trial, fs, band)
                    band_powers.append(power)
                avg_band_powers[band_name] = np.mean(band_powers, axis=0)
            
            # 绘制平均脑区功率对比
            if CLS:
                label_band_powers = {}
                unique_labels = np.unique(combined_labels)
                for label in unique_labels:
                    label_trials = combined_trials[combined_labels == label]
                    label_band_powers[label] = {}
                    for band_name, band in BANDS.items():
                        band_powers = []
                        for trial in label_trials:
                            power = calculate_band_power(trial, fs, band)
                            band_powers.append(power)
                        label_band_powers[label][band_name] = np.mean(band_powers, axis=0)
                plot_brain_region_comparison(label_band_powers, all_subjects_regions, output_prefix, combined_labels)
            else:
                plot_brain_region_comparison(avg_band_powers, all_subjects_regions, output_prefix)
        
    except Exception as e:
        print(f"处理过程中出错: {str(e)}")

    print("===== EEG频段功率可视化完成 =====")



