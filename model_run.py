import joblib

# --- 在另一个脚本中直接加载使用 ---
# 此时你甚至不需要加载原始的 CSV 数据
clf = joblib.load('rf_traffic_model.v1.pkl')

# 假设 new_data 是刚抓到的实时流量特征
# result = clf.predict(new_data)
