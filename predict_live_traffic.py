import pandas as pd
import numpy as np
import joblib
import os


# ==========================================
# 1. 特征对齐核心函数
# ==========================================
def align_nfstream_to_model(nfstream_df, keep_features):
    """
    将 NFStream 的特征格式转换为 XGBoost 模型所需的 CIC-IDS 格式
    """
    df = nfstream_df.copy()

    # --- 1. 直接映射关系 (NFStream -> CIC) ---
    mapping = {
        'src2dst_packets': 'Total Fwd Packet',
        'src2dst_bytes': 'Total Length of Fwd Packet',
        'dst2src_bytes': 'Total Length of Bwd Packet',
        'src2dst_max_ps': 'Fwd Packet Length Max',
        'src2dst_min_ps': 'Fwd Packet Length Min',
        'src2dst_mean_ps': 'Fwd Packet Length Mean',
        'src2dst_stddev_ps': 'Fwd Packet Length Std',
        'dst2src_max_ps': 'Bwd Packet Length Max',
        'dst2src_min_ps': 'Bwd Packet Length Min',
        'dst2src_mean_ps': 'Bwd Packet Length Mean',
        'bidirectional_min_ps': 'Packet Length Min',
        'bidirectional_max_ps': 'Packet Length Max',
        'bidirectional_mean_ps': 'Packet Length Mean',
        'bidirectional_stddev_ps': 'Packet Length Std',
        'bidirectional_psh_packets': 'PSH Flag Count',
        'bidirectional_ack_packets': 'ACK Flag Count'
    }
    df.rename(columns=mapping, inplace=True)

    # --- 2. 时间单位换算 (关键步！) ---
    # CIC-IDS 数据集的时间单位通常是微秒(us)，而 NFStream 是毫秒(ms)，需乘 1000
    time_mappings = {
        'bidirectional_duration_ms': 'Flow Duration',
        'bidirectional_mean_piat_ms': 'Flow IAT Mean',
        'bidirectional_stddev_piat_ms': 'Flow IAT Std',
        'bidirectional_max_piat_ms': 'Flow IAT Max',
        'bidirectional_min_piat_ms': 'Flow IAT Min',
        'src2dst_duration_ms': 'Fwd IAT Total',
        'src2dst_mean_piat_ms': 'Fwd IAT Mean',
        'src2dst_min_piat_ms': 'Fwd IAT Min',
        'dst2src_mean_piat_ms': 'Bwd IAT Mean',
        'dst2src_min_piat_ms': 'Bwd IAT Min'
    }
    for nf_col, cic_col in time_mappings.items():
        if nf_col in df.columns:
            df[cic_col] = df[nf_col] * 1000

    # --- 3. 统计衍生特征 ---
    # 避免除以 0 的情况，加上极小值 1e-5
    duration_sec = (df['Flow Duration'] / 1e6) + 1e-5
    df['Flow Bytes/s'] = (df['Total Length of Fwd Packet'] +
                          df['Total Length of Bwd Packet']) / duration_sec
    df['Flow Packets/s'] = (df['Total Fwd Packet'] +
                            nfstream_df['dst2src_packets']) / duration_sec
    df['Fwd Packets/s'] = df['Total Fwd Packet'] / duration_sec
    df['Bwd Packets/s'] = nfstream_df['dst2src_packets'] / duration_sec

    df['Packet Length Variance'] = df['Packet Length Std']**2
    df['Average Packet Size'] = df['Packet Length Mean']
    df['Fwd Segment Size Avg'] = df['Fwd Packet Length Mean']
    df['Bwd Segment Size Avg'] = df['Bwd Packet Length Mean']

    # 按照你的训练脚本，构造最终的两个高阶特征
    df['Flow_Density'] = df['Total Length of Fwd Packet'] / (
        df['Flow Duration'] + 1e-5)
    df['Packet_Size_Stability'] = df['Packet Length Std'] / (
        df['Packet Length Mean'] + 1e-5)

    # --- 4. 补全模型训练了，但 NFStream 没提取的特征 ---
    # 将缺失的偏门特征统一填充为 0
    for col in keep_features:
        if col not in df.columns:
            df[col] = 0

    return df


# ==========================================
# 2. 主预测流程
# ==========================================
if __name__ == "__main__":

    # 检查模型文件是否存在
    if not os.path.exists('final_xgboost_model.pkl') or not os.path.exists(
            'optimal_features_list.pkl'):
        print("[错误] 未找到模型文件或特征清单，请确保先运行了训练脚本。")
        exit()

    print("正在加载模型与特征清单...")
    model = joblib.load('final_xgboost_model.pkl')
    optimal_features = joblib.load('optimal_features_list.pkl')

    # 读取你通过 NFStream 提取出来的 csv 文件
    csv_file = r"D:\GraduationProject\src\evil.csv"
    print(f"正在读取实测流量数据: {csv_file}")
    df_raw = pd.read_csv(csv_file)

    # 提取展示所需的源/目的 IP 端口等基本信息（用于最后呈现结论）
    info_cols = ['src_ip', 'dst_ip', 'src_port', 'dst_port', 'protocol']
    df_info = df_raw[info_cols].copy()

    # 提取你在训练脚本里定义的 keep_features 清单
    keep_features = [
        'Flow Duration', 'Total Fwd Packet', 'Total Length of Fwd Packet',
        'Total Length of Bwd Packet', 'Fwd Packet Length Max',
        'Fwd Packet Length Min', 'Fwd Packet Length Mean',
        'Fwd Packet Length Std', 'Bwd Packet Length Max',
        'Bwd Packet Length Min', 'Bwd Packet Length Mean', 'Flow Bytes/s',
        'Flow Packets/s', 'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max',
        'Flow IAT Min', 'Fwd IAT Total', 'Fwd IAT Mean', 'Fwd IAT Min',
        'Bwd IAT Mean', 'Bwd IAT Min', 'Fwd Header Length', 'Fwd Packets/s',
        'Bwd Packets/s', 'Packet Length Min', 'Packet Length Max',
        'Packet Length Mean', 'Packet Length Std', 'Packet Length Variance',
        'PSH Flag Count', 'ACK Flag Count', 'Average Packet Size',
        'Fwd Segment Size Avg', 'Bwd Segment Size Avg', 'Bwd Bytes/Bulk Avg',
        'Subflow Fwd Bytes', 'FWD Init Win Bytes', 'Bwd Init Win Bytes',
        'Fwd Seg Size Min'
    ]

    print("正在对齐特征并进行单位转换...")
    df_aligned = align_nfstream_to_model(df_raw, keep_features)

    # 严格按照最优特征列表（optimal_features）筛选列
    # 这确保了喂给模型的列顺序与数量和训练时 100% 一致
    X_predict = df_aligned[optimal_features]

    # 填补可能出现的 NaN 或 Infinity
    X_predict = X_predict.replace([np.inf, -np.inf], np.nan).fillna(0)

    print("正在执行预测...")
    predictions = model.predict(X_predict)
    probabilities = model.predict_proba(X_predict)[:, 1]  # 提取判定为攻击的概率

    # 将预测结果合并回展示信息中
    df_info['Prediction'] = predictions
    df_info['Attack_Probability'] = probabilities.round(4)

    print("\n================== 预测结果报告 ==================")
    # 统计分类结果
    malicious_count = sum(predictions == 1)
    normal_count = sum(predictions == 0)
    print(f"总检测流量: {len(predictions)} 条")
    print(f"正常流量 (0): {normal_count} 条")
    print(f"攻击流量 (1): {malicious_count} 条\n")

    print("详细检测清单 (前 15 条):")
    # 如果输出太长，可以用 to_markdown() 打印出漂亮的表格
    print(df_info.head(15).to_markdown(index=False))
