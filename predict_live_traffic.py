import pandas as pd
import numpy as np
import joblib
import os


# 保留你原来的 align_nfstream_to_model 函数不变
def align_nfstream_to_model(nfstream_df, keep_features):
    # ... (此处保留你原有的对齐逻辑代码，保持不变) ...
    df = nfstream_df.copy()
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
    df['Flow_Density'] = df['Total Length of Fwd Packet'] / (
        df['Flow Duration'] + 1e-5)
    df['Packet_Size_Stability'] = df['Packet Length Std'] / (
        df['Packet Length Mean'] + 1e-5)
    for col in keep_features:
        if col not in df.columns:
            df[col] = 0
    return df


def predict_traffic(csv_file):
    """
    接收 CSV 路径，加载模型进行预测，返回带有判定结果的 DataFrame
    """
    model = joblib.load('final_xgboost_model.pkl')
    optimal_features = joblib.load('optimal_features_list.pkl')

    df_raw = pd.read_csv(csv_file)
    info_cols = ['src_ip', 'dst_ip', 'src_port', 'dst_port', 'protocol']
    df_info = df_raw[info_cols].copy()

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

    df_aligned = align_nfstream_to_model(df_raw, keep_features)
    X_predict = df_aligned[optimal_features].replace([np.inf, -np.inf],
                                                     np.nan).fillna(0)

    predictions = model.predict(X_predict)
    probabilities = model.predict_proba(X_predict)[:, 1]

    df_info['Prediction'] = predictions
    df_info['Attack_Probability'] = probabilities.round(4)
    # 将 0/1 转换为更直观的文本
    df_info['Status'] = df_info['Prediction'].apply(lambda x: '🚨 恶意攻击'
                                                    if x == 1 else '✅ 正常流量')

    return df_info


if __name__ == "__main__":
    # 本地测试代码
    res = predict_traffic("evil2.csv")
    print(res.head())
