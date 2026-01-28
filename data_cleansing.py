import pandas as pd
import numpy as np
import glob
import os


def load_and_clean_batch(file_list, label_value):
    all_chunks = []
    # 流量分析中通常不需要的特征（IP、端口等字符串列）
    drop_cols = [
        'Flow ID', 'Src IP', 'Src Port', 'Dst IP', 'Dst Port', 'Timestamp',
        'Malware_Sample'
    ]

    for f in file_list:
        if os.path.getsize(f) == 0: continue
        print(f"清洗中: {f}")
        try:
            # 分批读取，每批 50,000 行以节省内存
            reader = pd.read_csv(f, chunksize=50000)
            for chunk in reader:
                # 1. 扔掉无关列
                chunk.drop(
                    columns=[col for col in drop_cols if col in chunk.columns],
                    inplace=True)

                # 2. 贴上数字标签
                chunk['Label'] = label_value

                # 3. 内存优化：转为 float32 (比 float64 节省一半内存)
                cols = chunk.columns.drop('Label')
                chunk[cols] = chunk[cols].apply(
                    pd.to_numeric, errors='coerce').astype('float32')

                # 4. 填充无效值
                chunk.replace([np.inf, -np.inf], np.nan, inplace=True)
                chunk.fillna(0, inplace=True)

                all_chunks.append(chunk)
        except Exception as e:
            print(f"跳过异常文件 {f}: {e}")

    return pd.concat(all_chunks, ignore_index=True)


# 执行批量操作
mal_files = glob.glob(
    "./src/Malicious-CSVs/Malicious-CSVs/Network-CSVs/FinalVersion-PerSample/**/*.csv",
    recursive=True)
ben_files = glob.glob("./src/Benign-Network-CSVs/Network-CSVs/**/*.csv",
                      recursive=True)

df_mal = load_and_clean_batch(mal_files, 1)
df_ben = load_and_clean_batch(ben_files, 0)

# 合并并保存为 Parquet (比 CSV 读取快 10 倍)
df_total = pd.concat([df_mal, df_ben], ignore_index=True)
df_total.to_parquet("processed_data.parquet", index=False)
print(f"\n清洗完毕！共处理 {len(df_total)} 行数据，已保存至 processed_data.parquet")
