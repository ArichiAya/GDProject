import pandas as pd
import numpy as np
import glob
import joblib
import os
import xgboost as xgb  # 导入 XGBoost
from sklearn.model_selection import train_test_split


def load_and_clean_batch(file_list, label_value):
    all_chunks = []
    drop_cols = [
        'Flow ID', 'Src IP', 'Src Port', 'Dst IP', 'Dst Port', 'Timestamp',
        'Malware_Sample'
    ]

    for f in file_list:
        if os.path.getsize(f) == 0: continue
        print(f"正在处理文件: {f}")
        try:
            # 增加 engine='pyarrow' 可以提速读取
            reader = pd.read_csv(f, chunksize=50000)
            for chunk in reader:
                chunk.drop(
                    columns=[col for col in drop_cols if col in chunk.columns],
                    inplace=True)
                chunk['Label'] = label_value
                cols = chunk.columns.drop('Label')
                chunk[cols] = chunk[cols].apply(
                    pd.to_numeric, errors='coerce').astype('float32')
                chunk.replace([np.inf, -np.inf], np.nan, inplace=True)
                chunk.fillna(0, inplace=True)
                all_chunks.append(chunk)
        except Exception as e:
            print(f"跳过坏文件 {f}: {e}")

    return pd.concat(all_chunks, ignore_index=True)


# --- 1. 加载数据 (同之前逻辑) ---
mal_files = glob.glob(
    "./src/Malicious-CSVs/Malicious-CSVs/Network-CSVs/FinalVersion-PerSample/**/*.csv",
    recursive=True)
ben_files = glob.glob("./src/Benign-Network-CSVs/Network-CSVs/**/*.csv",
                      recursive=True)

df_mal = load_and_clean_batch(mal_files, 1)
df_ben = load_and_clean_batch(ben_files, 0)

if len(df_ben) > len(df_mal):
    df_ben = df_ben.sample(n=len(df_mal), random_state=42)

df_total = pd.concat([df_mal, df_ben], ignore_index=True)
del df_mal, df_ben

X = df_total.drop(columns=['Label'])
y = df_total['Label']

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=42)

# --- 2. 核心：配置 XGBoost GPU 训练 ---
print("\n🚀 启动 XGBoost GPU 加速训练...")

model = xgb.XGBClassifier(
    # 关键参数：使用 GPU 进行直方图加速
    tree_method='gpu_hist',
    gpu_id=0,  # 使用第一块显卡
    n_estimators=100,  # 树的数量
    max_depth=6,  # 树的最大深度
    learning_rate=0.1,  # 学习率
    verbosity=1  # 显示训练进度
)

model.fit(X_train, y_train)

# --- 3. 保存模型与特征分析 ---
joblib.dump(model, 'xgboost_traffic_gpu_model.joblib')

# XGBoost 的特征重要性获取方式略有不同
importances = pd.Series(model.feature_importances_,
                        index=X.columns).sort_values(ascending=False)
print("\n--- XGBoost 筛选的关键特征 ---")
print(importances.head(15))
