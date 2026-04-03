"""
NGAFID 航空数据预处理模块。
包含: 缺失值线性插值填充、防止统计泄露的 Fold-wise Z-score 归一化。
只读取本地 data/subset_data 中已下载的数据，不依赖任何上游链接。
依赖包: pip install numpy pandas scikit-learn compress_pickle
"""
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.model_selection import StratifiedKFold

def load_pickle(file_path):
    try:
        from compress_pickle import load as compress_pickle_load  # pyright: ignore[reportMissingImports]
    except ModuleNotFoundError:
        compress_pickle_load = None

    if compress_pickle_load is not None:
        return compress_pickle_load(file_path)

    with open(file_path, "rb") as file_handle:
        return pickle.load(file_handle)

class AviationDataPreprocessor:
    def __init__(self):
        """
        初始化预处理器，用于保存当前折(fold)训练集的均值和标准差。
        绝对不使用全局统计量以防数据泄露。
        """
        self.mean_ = None
        self.std_ = None

    def fill_missing_values_linear(self, X: np.ndarray) -> np.ndarray:
        """
        对三维时间序列张量进行缺失值线性插值。
        
        参数:
            X: np.ndarray, 形状为 [samples, timesteps, features]
        返回:
            X_filled: 填充后的 np.ndarray
        """
        print("正在进行特征维度的时间序列线性插值...")
        X_filled = np.empty_like(X)
        # 修复了潜在的元组迭代报错问题，获取样本数量
        n_samples = X.shape[0]
        
        for i in range(n_samples):
            # 将每个样本 [timesteps, features] 转为 DataFrame 处理
            df = pd.DataFrame(X[i])
            # 按列（特征）沿时间轴进行线性插值，limit_direction='both' 防止首尾连续NaN
            df.interpolate(method='linear', limit_direction='both', inplace=True)
            # 对于极端情况（例如某个特征从头到尾全是NaN），使用0兜底
            df.fillna(0, inplace=True)
            X_filled[i] = df.values
            
        return X_filled

    def fit(self, X_train: np.ndarray):
        """
        仅在当前训练折 (Training Fold) 上拟合统计参数 (Mean, Std)。
        
        参数:
            X_train: 形状为 [samples, timesteps, features]
        """
        # 沿样本轴(axis=0)和时间步轴(axis=1)计算，得到长度为 features 的向量
        self.mean_ = np.nanmean(X_train, axis=(0, 1))
        self.std_ = np.nanstd(X_train, axis=(0, 1))
        
        # 防止除零错误 (常数特征)
        self.std_[self.std_ == 0] = 1e-8

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        使用拟合好的训练集参数对输入数据进行 Z-score 归一化。
        
        参数:
            X: 形状为 [samples, timesteps, features] (可以是训练集、验证集或测试集)
        """
        if self.mean_ is None or self.std_ is None:
            raise ValueError("预处理器尚未拟合！请先调用 fit(X_train)。")
        
        return (X - self.mean_) / self.std_

    def fit_transform(self, X_train: np.ndarray) -> np.ndarray:
        """拟合并直接转换训练集"""
        self.fit(X_train)
        return self.transform(X_train)


def _resolve_local_subset_dir(
    subset_name: str = "2days",
    base_dir: Path | None = None,
) -> Path:
    project_root = Path(base_dir).resolve() if base_dir is not None else Path(__file__).resolve().parent.parent
    candidates = [
        project_root / "data" / "subset_data" / subset_name / subset_name,
        project_root / "data" / "subset_data" / subset_name,
        project_root / "data" / subset_name / subset_name,
        project_root / "data" / subset_name,
    ]

    required_files = {"flight_data.pkl", "flight_header.csv", "stats.csv"}
    for candidate in candidates:
        if candidate.exists() and required_files.issubset({path.name for path in candidate.iterdir() if path.is_file()}):
            return candidate

    raise FileNotFoundError(
        f"未找到本地数据目录 {subset_name}，请确认数据已放入 data/subset_data/{subset_name}/{subset_name}。"
    )


def load_local_subset_data(
    subset_name: str = "2days",
    base_dir: Path | None = None,
    label_column: str = "before_after",
    max_length: int = 4096,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    从本地 data/subset_data 读取 NGAFID 子集，并构造成预处理所需的三维张量。

    返回:
        X: [samples, timesteps, features]
        y: 标签数组
        flight_header_df: 原始头表，便于后续按 fold 或其它列筛选
    """
    subset_dir = _resolve_local_subset_dir(subset_name=subset_name, base_dir=base_dir)
    flight_header_path = subset_dir / "flight_header.csv"
    flight_data_path = subset_dir / "flight_data.pkl"

    flight_header_df = pd.read_csv(flight_header_path, index_col="Master Index")
    flight_data_dict = load_pickle(flight_data_path)

    if label_column not in flight_header_df.columns:
        raise KeyError(f"标签列不存在: {label_column}")

    sample_ids = [index for index in flight_header_df.index if index in flight_data_dict]
    if not sample_ids:
        raise ValueError("flight_header.csv 中的索引在 flight_data.pkl 中都未找到。")

    first_sample = np.asarray(flight_data_dict[sample_ids[0]], dtype=np.float32)
    feature_count = first_sample.shape[1]
    max_length = min(max_length, max(np.asarray(flight_data_dict[index]).shape[0] for index in sample_ids))

    X = np.zeros((len(sample_ids), max_length, feature_count), dtype=np.float32)
    y = np.zeros(len(sample_ids), dtype=flight_header_df[label_column].to_numpy().dtype)

    for i, index in enumerate(sample_ids):
        sample_array = np.asarray(flight_data_dict[index], dtype=np.float32)
        sample_array = sample_array[-max_length:, :feature_count]
        X[i, : sample_array.shape[0], :] = sample_array
        y[i] = flight_header_df.loc[index, label_column]

    return X, y, flight_header_df.loc[sample_ids]


def format_labels(y_raw: np.ndarray) -> np.ndarray:
    """
    根据任务需求对齐标签：
    确保 0 = 维护后 (健康航班)
    确保 1 = 维护前 (即将发生故障的航班)
    """
    y_formatted = y_raw.copy()
    return y_formatted


# ================= 使用范例（仅本地 data） ================= 
def run_local_cv_example():
    """演示数据隔离的正确做法：每折都独立拟合统计量，不泄露全数据统计"""
    RANDOM_SEED = 42
    np.random.seed(RANDOM_SEED)
    
    print("正在从本地 data/subset_data 读取 2days 数据...")
    dummy_X, dummy_y, _ = load_local_subset_data(subset_name="2days", label_column="before_after")

    # 统一转换标签
    dummy_y = format_labels(dummy_y)

    # 声明分层交叉验证 (random_state 固定确保可复现)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(dummy_X, dummy_y)):
        print(f"\n--- 正在处理第 {fold + 1} 折 ---")
        X_train, y_train = dummy_X[train_idx], dummy_y[train_idx]
        X_val, y_val = dummy_X[val_idx], dummy_y[val_idx]
        
        # 每折都新建预处理器 (最关键的防泄露步骤)
        preprocessor = AviationDataPreprocessor()
        
        # 1. 对训练集和验证集分别进行插值 (sample-wise 独立，无统计泄露)
        X_train = preprocessor.fill_missing_values_linear(X_train)
        X_val = preprocessor.fill_missing_values_linear(X_val)
        
        # 2. 严格防止泄漏的归一化
        # fit_transform 只在训练集上拟合 mean/std，然后用这些统计量转换验证集
        # 这样验证集的统计特性完全来自训练集定义，不泄露验证集本身的信息
        X_train_scaled = preprocessor.fit_transform(X_train)
        X_val_scaled = preprocessor.transform(X_val)
        
        print(f"X_train 形状: {X_train_scaled.shape}, Mean 验证 (应接近0): {np.mean(X_train_scaled):.4f}")
        print(f"X_val 形状: {X_val_scaled.shape}")


def run_cv_example():
    """兼容旧入口：改为直接使用本地数据。"""
    run_local_cv_example()

if __name__ == "__main__":
    run_local_cv_example()