import pandas as pd
from sklearn.model_selection import KFold, train_test_split
import os

def create_kfold_splits(dataset_name: str, file_path: str, n_splits: int = 5, random_seed: int = 42):
    """
    加载数据集，并使用 5 折交叉验证。
    初始划分为 初级训练集(4):测试集(1)。
    再从初级训练集中划分出 0.125 (即 1/8) 作为验证集。
    最终比例为 训练集(70%):验证集(10%):测试集(20%)，即 7:1:2 划分。
    """
    print(f"--- 开始处理数据集: {dataset_name.upper()} ({n_splits}折交叉验证) ---")

    # 加载数据
    try:
        df = pd.read_csv(file_path)
        print(f"成功加载数据集: {file_path} | 形状: {df.shape}")
    except FileNotFoundError:
        print(f"错误: 文件未找到 {file_path}。请确保文件路径正确。")
        return

    # 初始化 KFold (n_splits=5 意味着 4/5 为初级训练集，1/5 为测试集)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)

    for fold, (train_idx, test_idx) in enumerate(kf.split(df)):
        # 为每个 fold 创建独立的输出目录
        fold_dir = os.path.join("dataset", "splits_dataset", dataset_name, f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)

        # 1. 按照 4:1 划分出初级训练集和测试集 (测试集独立，保持不变)
        train_temp_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]

        # 2. 将初级训练集中的 0.125 划分为验证集，剩下的 0.875 作为最终训练集
        train_df, val_df = train_test_split(train_temp_df, test_size=0.125, random_state=random_seed, shuffle=True)

        print(f"Fold {fold} 划分完成 (7:1:2) -> 训练集: {len(train_df)}, 验证集: {len(val_df)}, 测试集: {len(test_df)}")

        # 保存为CSV文件
        train_df.to_csv(os.path.join(fold_dir, "train.csv"), index=False)
        val_df.to_csv(os.path.join(fold_dir, "val.csv"), index=False)
        test_df.to_csv(os.path.join(fold_dir, "test.csv"), index=False)

    print(f"数据已成功保存至 'dataset/splits_dataset/{dataset_name}' 文件夹中。\n" + "-" * 50)


if __name__ == '__main__':
    # 定义要处理的数据集 (请确保这里的路径是你本地真实的 CSV 路径)
    datasets_to_process = {
        'davis': r'./dataset\Davis.csv',
        'kiba': r'./dataset\KIBA.csv'
    }

    # 调用时更新了 n_splits=5 以符合 4:1 的要求
    for name, path in datasets_to_process.items():
        create_kfold_splits(dataset_name=name, file_path=path, n_splits=5)