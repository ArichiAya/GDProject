import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# 1. 快速加载清洗后的数据
print("正在从 Parquet 加载预处理数据...")
df = pd.read_parquet("processed_data.parquet")

# 2. 样本均衡处理 (如果正常包远多于恶意包)
mal_count = df[df['Label'] == 1].shape[0]
ben_df = df[df['Label'] == 0].sample(n=mal_count, random_state=42)
mal_df = df[df['Label'] == 1]
df_balanced = pd.concat([ben_df, mal_df])

# 3. 划分特征与标签
X = df_balanced.drop(columns=['Label'])
y = df_balanced['Label']
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=42)

# 4. 训练随机森林 (n_jobs=-1 跑满所有 CPU 核心)
print(f"启动随机森林训练 (训练集规模: {X_train.shape})...")
rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
rf.fit(X_train, y_train)

# 5. 评估结果
y_pred = rf.predict(X_test)
print("\n--- 训练评估报告 ---")
print(classification_report(y_test, y_pred))

# 6. 保存模型 (存档)
joblib.dump(rf, 'random_forest_traffic_model.joblib')
print("模型已保存为 random_forest_traffic_model.joblib")

# 7. 特征重要性分析
importances = pd.Series(rf.feature_importances_,
                        index=X.columns).sort_values(ascending=False)
print("\n--- 关键特征排行榜 (Top 10) ---")
print(importances.head(10))
