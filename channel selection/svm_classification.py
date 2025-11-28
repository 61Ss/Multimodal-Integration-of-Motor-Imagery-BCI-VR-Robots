import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 配置参数
DATA_FILE = 'e:\surf_selection\data_set_IVa_aw_selected.mat'
TEST_SIZE = 0.3  # 测试集比例
RANDOM_STATE = 42  # 随机种子，保证结果可复现

class SVMClassifier:
    def __init__(self):
        self.data = None
        self.labels = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.pca = None
        self.model = None

    def load_data(self):
        """加载筛选后的数据"""
        if not os.path.exists(DATA_FILE):
            raise FileNotFoundError(f"文件 {DATA_FILE} 不存在")

        mat_data = loadmat(DATA_FILE)
        self.data = mat_data['trials']  # 形状: (num_trials, time_points, channels)
        self.labels = mat_data['trial_labels']  # 形状: (num_trials,)

        # 确保labels是一维数组
        if self.labels.ndim > 1:
            self.labels = self.labels.flatten()

        # 修正标签：将1和2变为0和1
        self.labels = self.labels.astype(int) - 1

        print(f"成功加载数据，形状: {self.data.shape}")
        print(f"标签数量: {len(self.labels)}")
        print(f"类别分布: {np.bincount(self.labels)}")

        return self

    def preprocess_data(self, n_components=0.95):
        """数据预处理：降维和标准化"""
        # 重塑数据: (num_trials, time_points * channels)
        num_trials, num_time_points, num_channels = self.data.shape
        X = self.data.reshape(num_trials, -1)

        # 应用PCA降维
        self.pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
        X_pca = self.pca.fit_transform(X)
        print(f"PCA降维后的数据形状: {X_pca.shape}")
        print(f"保留的方差比例: {self.pca.explained_variance_ratio_.sum():.4f}")

        # 数据标准化
        X_scaled = self.scaler.fit_transform(X_pca)

        # 划分训练集和测试集
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_scaled, self.labels, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=self.labels
        )

        print(f"训练集大小: {self.X_train.shape}")
        print(f"测试集大小: {self.X_test.shape}")

        return self

    def train_model(self, param_grid=None):
        """训练SVM模型"""
        if param_grid is None:
            # 默认参数网格
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.1, 1],
                'kernel': ['linear', 'rbf', 'poly']
            }

        # 使用网格搜索寻找最佳参数
        grid_search = GridSearchCV(SVC(random_state=RANDOM_STATE), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(self.X_train, self.y_train)

        # 获取最佳模型
        self.model = grid_search.best_estimator_
        print(f"最佳参数: {grid_search.best_params_}")
        print(f"交叉验证准确率: {grid_search.best_score_:.4f}")

        return self

    def evaluate_model(self):
        """评估模型性能"""
        # 在测试集上进行预测
        y_pred = self.model.predict(self.X_test)

        # 计算准确率
        accuracy = accuracy_score(self.y_test, y_pred)
        print(f"测试集准确率: {accuracy:.4f}")

        # 混淆矩阵
        cm = confusion_matrix(self.y_test, y_pred)
        print("混淆矩阵:")
        print(cm)

        # 分类报告
        print("分类报告:")
        print(classification_report(self.y_test, y_pred))

        # 可视化混淆矩阵
        self._plot_confusion_matrix(cm)

        return accuracy

    def _plot_confusion_matrix(self, cm):
        """可视化混淆矩阵"""
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('混淆矩阵')
        plt.colorbar()
        tick_marks = np.arange(len(np.unique(self.labels)))
        plt.xticks(tick_marks, np.unique(self.labels))
        plt.yticks(tick_marks, np.unique(self.labels))

        # 在矩阵中显示数值
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('真实标签')
        plt.xlabel('预测标签')
        plt.tight_layout()
        plt.savefig('svm_confusion_matrix.png')
        print("混淆矩阵图已保存为 svm_confusion_matrix.png")
        plt.close()

if __name__ == '__main__':
    # 创建分类器实例
    classifier = SVMClassifier()

    try:
        # 加载数据
        classifier.load_data()

        # 预处理数据
        classifier.preprocess_data(n_components=0.95)

        # 训练模型
        classifier.train_model()

        # 评估模型
        classifier.evaluate_model()

    except Exception as e:
        print(f"处理过程中出错: {e}")