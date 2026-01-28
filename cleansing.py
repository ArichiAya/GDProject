import pandas as pd
import numpy as np
import glob
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def load_and_clean_batch(file_list, label_value):
    all_chunks = []
    drop_cols = [
        'Flow ID', 'Src IP', 'Src Port', 'Dst IP', 'Dst Port', 'Timestamp',
        'Malware_Sample'
    ]

    for f in file_list:
        # --- 防线 1: 检查文件物理大小 ---
        if os.path.getsize(f) == 0:
            print(f"跳过空文件 (0字节): {f}")
            continue

        print(f"正在处理文件: {f}")

        # --- 防线 2: 捕获读取时的异常 ---
        try:
            reader = pd.read_csv(f, chunksize=50000)

            # 记录该文件是否成功提供了数据
            file_has_data = False

            for chunk in reader:
                file_has_data = True
                # 立即剔除无关列
                chunk.drop(
                    columns=[col for col in drop_cols if col in chunk.columns],
                    inplace=True)

                # 贴标签
                chunk['Label'] = label_value

                # 内存优化与类型转换
                cols = chunk.columns.drop('Label')
                chunk[cols] = chunk[cols].apply(
                    pd.to_numeric, errors='coerce').astype('float32')

                chunk.replace([np.inf, -np.inf], np.nan, inplace=True)
                chunk.fillna(0, inplace=True)

                all_chunks.append(chunk)

            if not file_has_data:
                print(f"警告: 文件有表头但无内容: {f}")

        except pd.errors.EmptyDataError:
            print(f"跳过异常文件 (EmptyDataError): {f}")
        except Exception as e:
            print(f"跳过文件 {f}，原因: {e}")

    return pd.concat(all_chunks,
                     ignore_index=True) if all_chunks else pd.DataFrame()


# --- 1. 使用 glob 批量搜寻路径 ---
# **/*.csv 表示搜索当前文件夹及所有子文件夹下的 CSV
mal_files = glob.glob(
    "./src/Malicious-CSVs/Malicious-CSVs/Network-CSVs/FinalVersion-PerSample/**/*.csv",
    recursive=True)
ben_files = glob.glob("./src/Benign-Network-CSVs/Network-CSVs/**/*.csv",
                      recursive=True)

# --- 2. 执行批量加载 ---
print("开始加载恶意流量...")
df_mal = load_and_clean_batch(mal_files, 1)

print("开始加载正常流量...")
df_ben = load_and_clean_batch(ben_files, 0)

# --- 3. 样本均衡处理 ---
if len(df_ben) > len(df_mal):
    print(f"正常样本({len(df_ben)})多于恶意样本({len(df_mal)})，进行随机下采样...")
    df_ben = df_ben.sample(n=len(df_mal), random_state=42)

# 合并总表并释放临时内存
df_total = pd.concat([df_mal, df_ben], ignore_index=True)
del df_mal, df_ben

# --- 4. 训练模型 ---
X = df_total.drop(columns=['Label'])
y = df_total['Label']

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=42)

print(f"训练集规模: {X_train.shape}")
model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
model.fit(X_train, y_train)

# --- 5. 使用 joblib 保存模型 ---
model_filename = 'traffic_rf_model_v1.joblib'
joblib.dump(model, model_filename)
print(f"--- 模型已保存至: {model_filename} ---")

# --- 6. 输出特征重要性 ---
importances = pd.Series(model.feature_importances_,
                        index=X.columns).sort_values(ascending=False)
print("\n--- 关键特征排行榜 ---")
print(importances.head(15))
