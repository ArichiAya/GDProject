import pandas as pd
import joblib

# 1. 读取刚刚提取的真实流量特征
live_data = pd.read_csv("live_features.csv")

# 2. 列名映射字典 (这里列举几个核心的，你需要根据你最终的 optimal_features 补全)
# 左边是 cicflowmeter 生成的列名，右边是你模型需要的列名
rename_map = {
    'flow_duration': 'Flow Duration',
    'tot_fwd_pkts': 'Total Fwd Packet',
    'totlen_fwd_pkts': 'Total Length of Fwd Packet',
    'totlen_bwd_pkts': 'Total Length of Bwd Packet',
    'fwd_pkt_len_max': 'Fwd Packet Length Max',
    'fwd_pkt_len_min': 'Fwd Packet Length Min',
    'fwd_pkt_len_mean': 'Fwd Packet Length Mean',
    'fwd_pkt_len_std': 'Fwd Packet Length Std',
    'flow_byts_s': 'Flow Bytes/s',
    'flow_pkts_s': 'Flow Packets/s',
    'pkt_len_std': 'Packet Length Std',
    'pkt_len_mean': 'Packet Length Mean'
    # ... 请根据你的 optimal_features 列表对应检查补全
}

# 3. 重命名列
live_data = live_data.rename(columns=rename_map)

# 4. 构造衍生特征 (跟你训练时一模一样)
live_data['Flow_Density'] = live_data['Total Length of Fwd Packet'] / (
    live_data['Flow Duration'] + 1e-5)
live_data['Packet_Size_Stability'] = live_data['Packet Length Std'] / (
    live_data['Packet Length Mean'] + 1e-5)

# 5. 加载模型和特征列表进行预测
model = joblib.load('final_xgboost_model.pkl')
optimal_features = joblib.load('optimal_features_list.pkl')

# 提取需要的特征并处理可能出现的 NaN (缺失值)
X_live = live_data[optimal_features].fillna(0)

# 6. 见证奇迹的时刻
predictions = model.predict(X_live)
live_data['Is_Attack'] = predictions

# 打印出被判定为攻击的流 (如果全是正常流量，应该输出为空或极少)
print("发现异常流量的条数:", sum(predictions == 1))
print(live_data[live_data['Is_Attack'] == 1][[
    'src_ip', 'dst_ip', 'dst_port', 'Is_Attack'
]])
