import os

import numpy as np
from sklearn.decomposition import IncrementalPCA
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from scipy.io import loadmat
from scipy.signal import welch
import mne


# ---------------------- 数据加载 ----------------------
def load_eeg_data(subject_id, input_dir='100cc'):
    """加载处理后的EEG数据，包括电极坐标和trial标签"""
    filename = f'data_set_IVa_{subject_id}.mat'
    file_path = os.path.join(input_dir, filename)

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
    x_pos = info['xpos'].flatten() * 0.1  # 应用坐标缩放
    y_pos = info['ypos'].flatten() * 0.1

    # 确定每个电极的脑区
    regions = get_brain_regions(ch_names)

    # 创建电极位置字典 (MNE需要3D坐标，z设为0)
    ch_pos = {ch_names[i]: [x_pos[i], y_pos[i], 0.0] for i in range(len(ch_names))}

    return trials, fs, ch_names, ch_pos, trial_labels, regions


def get_brain_regions(ch_names):
    """根据电极名称前缀确定每个电极的脑区"""
    region_map = {
        'F': 'Frontal', 'C': 'Central', 'P': 'Parietal', 'O': 'Occipital', 'T': 'Temporal'
    }
    return [region_map.get(ch[0], 'Other') for ch in ch_names]


# ---------------------- 频段功率计算 ----------------------
# def calculate_band_power(data, fs, band):
#     """计算指定频段的功率谱密度 (PSD)"""
#     fmin, fmax = band
#     freqs, psd = welch(data, fs=fs, nperseg=128, noverlap=0, axis=0)
#     print(psd.shape)
#     freq_mask = (freqs >= fmin) & (freqs <= fmax)
#     band_power = np.trapz(psd[freq_mask, :], freqs[freq_mask], axis=0)
#     return band_power

# def calculate_band_power(data, fs, band, nperseg=128, noverlap=64):
#     """
#     计算指定频段的功率谱密度 (PSD)，手动合并时间点以获得三维 psd
#     data：形状为 (num_samples, num_channels) 的二维数组
#     fs：采样率
#     band：频带范围 (fmin, fmax)，例如 (7, 13) 为 alpha 波
#     nperseg：每个窗口的大小
#     noverlap：窗口的重叠部分
#
#     返回:
#     band_power：频带功率，形状为 (num_timepoints, num_channels)，表示每个时间窗口每个通道的频带功率
#     """
#     fmin, fmax = band
#     num_samples, num_channels = data.shape
#
#     # 计算时间窗口的数量 (num_timepoints)，根据滑动窗口计算
#     num_timepoints = (num_samples - nperseg) // (nperseg - noverlap) + 1
#
#     # 初始化一个空数组来保存每个时间窗口的psd，形状是 (num_frequencies, num_timepoints, num_channels)
#     psd_all_windows = np.zeros((nperseg // 2 + 1, num_timepoints, num_channels))  # 每个窗口的频率点数
#
#     # 对每个时间窗口进行计算，合并得到三维的 psd
#     for t in range(num_timepoints):
#         # 计算当前窗口的开始和结束位置
#         start = t * (nperseg - noverlap)
#         end = start + nperseg
#
#         # 截取当前窗口的信号
#         window_data = data[start:end, :]
#
#         # 使用 welch 计算当前窗口的 psd
#         freqs, psd = welch(window_data, fs=fs, nperseg=nperseg, noverlap=noverlap, axis=0)
#
#         # 存储当前窗口的 psd
#         psd_all_windows[:, t, :] = psd
#
#     # 使用频率掩码获取目标频段的数据
#     freq_mask = (freqs >= fmin) & (freqs <= fmax)
#
#     # 对每个时间窗口的频谱进行积分，得到每个时间窗口的功率
#     # 结果的形状是 (num_timepoints, num_channels)
#     band_power = np.trapz(psd_all_windows[freq_mask, :, :], freqs[freq_mask], axis=0)
#
#     # 确保 band_power 是二维的，形状是 (num_timepoints, num_channels)
#     if band_power.ndim == 1:
#         band_power = band_power[:, np.newaxis]  # 如果是1维数据，转为二维数据，列为通道
#
#     # 打印 band_power 的形状和内容，确保它是二维的
#     # print(f"Band Power Shape: {band_power.shape}")  # 应该是 (num_timepoints, num_channels)
#     # print(f"Band Power Values: {band_power}")  # 打印 band_power 的值
#
#     return band_power

from scipy.signal import spectrogram


def calculate_band_power(data, fs, band, nperseg=128, noverlap=64):
    fmin, fmax = band
    num_samples, num_channels = data.shape

    # 使用 spectrogram 自动处理每个时间窗口的 PSD 计算
    freqs, times, Sxx = spectrogram(data, fs=fs, nperseg=nperseg, noverlap=noverlap)
    print(Sxx.shape)  # 打印 Sxx 的形状，查看其维度

    # 使用频率掩码提取目标频段
    freq_mask = (freqs >= fmin) & (freqs <= fmax)

    # 对频段内的频谱积分得到每个时间窗口的功率
    # 在这里，需要沿着频率维度（Sxx的第一个维度）应用频率掩码
    band_power = np.trapz(Sxx[freq_mask, :, :], freqs[freq_mask], axis=0)

    # 确保返回的是二维形状（num_timepoints, num_channels）
    if band_power.ndim == 1:
        band_power = band_power[:, np.newaxis]

    return band_power


# ---------------------- 互信息计算 ----------------------
def calculate_mutual_information(trial_data, fs, ch_names):
    """
    计算每个通道与其他所有通道的互信息 (alpha, beta, gamma)
    """
    band_ranges = {'alpha': (7, 13), 'beta': (14, 30), 'gamma': (30, 50)}
    mi_dict = {'alpha': [], 'beta': [], 'gamma': []}

    for band, (fmin, fmax) in band_ranges.items():
        # 计算该频段的功率
        band_power = calculate_band_power(trial_data, fs, (fmin, fmax))

        # 确保 band_power 是二维的（num_timepoints, num_channels）
        X = band_power.T  # 转置，使每列为一个通道
        num_channels = X.shape[1]
        if num_channels < 2:
            print(f"Skipping mutual information calculation for {band} band as there are fewer than 2 channels.")
            continue

        mi_matrix = np.zeros((num_channels, num_channels))

        # 计算每对通道之间的互信息
        for i in range(num_channels):
            for j in range(i + 1, num_channels):  # 计算对称矩阵时，避免重复计算
                X_pair = np.column_stack((X[:, i], X[:, j]))

                # 使用 mutual_info_regression 计算每对通道的互信息
                mi = mutual_info_regression(X_pair, np.zeros(X_pair.shape[0]))  # 无标签，计算互信息
                mi_matrix[i, j] = mi[0]

        # 打印互信息矩阵，查看是否有异常
        print(f"Mutual Information Matrix for {band} band:")
        print(mi_matrix)

        mi_dict[band] = mi_matrix

    return mi_dict
# ---------------------- 计算密度 ----------------------
def calculate_density(trials, pca_results, k_neighbors=5):
    """计算每个trial的密度，基于K近邻"""
    nn = NearestNeighbors(n_neighbors=k_neighbors)
    nn.fit(pca_results)  # 使用PCA特征计算距离
    distances, _ = nn.kneighbors(pca_results)
    density = np.mean(distances, axis=1)  # 平均距离越低，密度越高
    return density


# ---------------------- 处理所有trial并选择前四大通道 ----------------------
def process_trials_and_select_top_channels(trials, trial_labels, fs, ch_names, pca_results):
    """
    处理所有的trial，计算每个频段的互信息，并选择前四大通道，并根据密度划分高低密度
    """
    top_channels = []

    # 计算密度
    density = calculate_density(trials, pca_results)

    for trial_idx, trial_data in enumerate(trials):
        # 计算当前trial的互信息
        mi_dict = calculate_mutual_information(trial_data, fs, ch_names)

        # 计算alpha、beta、gamma频段的平均互信息
        avg_mi = np.mean([mi_dict['alpha'], mi_dict['beta'], mi_dict['gamma']], axis=0)

        # 每个通道的互信息之和，并选择前四大的通道
        channel_sum_mi = np.sum(avg_mi, axis=1)  # 每个通道的总互信息
        top_channels_idx = np.argsort(channel_sum_mi)[-4:]  # 获取前四个通道

        # 获取前四大通道名称
        top_channel_names = [ch_names[i] for i in top_channels_idx]

        # 获取密度（低密度或高密度）
        trial_density = 'High' if density[trial_idx] > np.percentile(density, 85) else 'Low'

        # 输出trial的信息，包括label、密度和top 4 channels
        trial_result = {
            'trial_index': trial_idx,
            'label': trial_labels[trial_idx],
            'density': trial_density,
            'top_channels': top_channel_names
        }
        top_channels.append(trial_result)

    return top_channels


# ---------------------- 主程序 ----------------------
def main():
    subject_id = 'aw'  # 可以更改为你自己的受试者ID
    trials, fs, ch_names, ch_pos, trial_labels, regions = load_eeg_data(subject_id)

    # 使用增量PCA降维，用于计算密度
    n_components = 10  # 设置PCA组件数量为10，确保小于每个批次的大小
    ipca = IncrementalPCA(n_components=n_components, batch_size=10)  # 设置批次大小为10

    # 将trials展平成二维数据，并使用增量PCA逐步训练
    n_trials, n_timepoints, n_channels = trials.shape
    trials_reshaped = trials.reshape(n_trials, -1)  # 将trials展平成二维数据
    for batch_start in range(0, n_trials, 10):  # 每批次大小为10
        batch_data = trials_reshaped[batch_start:batch_start + 10]  # 取每批次数据
        ipca.partial_fit(batch_data)  # 更新PCA模型

    pca_results = ipca.transform(trials_reshaped)  # 使用增量PCA转换所有数据

    # 处理所有trial并选择前四大通道
    top_channels = process_trials_and_select_top_channels(trials, trial_labels, fs, ch_names, pca_results)

    # 输出结果
    for result in top_channels:
        print(
            f"Trial {result['trial_index']} - Label: {result['label']}, Density: {result['density']}, Top Channels: {result['top_channels']}")


if __name__ == '__main__':
    main()
