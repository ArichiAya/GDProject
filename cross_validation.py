import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_validate, KFold
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 快速加载清洗后的数据
print("正在从 Parquet 加载预处理数据...")
df = pd.read_parquet("processed_data.parquet")

# 2. 样本均衡处理
# 分别提取恶意和正常样本
mal_df = df[df['Label'] == 1]
ben_df = df[df['Label'] == 0]

mal_count = len(mal_df)
ben_count = len(ben_df)

print(f"原始数据统计：恶意样本 {mal_count} 条，正常样本 {ben_count} 条")

if ben_count > mal_count:
    # 如果正常包多，则把正常包减少到和恶意包一样多
    print("采取下采样：缩减正常样本数量...")
    ben_df = ben_df.sample(n=mal_count, random_state=42)
elif mal_count > ben_count:
    # 如果恶意包反而多，则把恶意包减少到和正常包一样多
    print("采取下采样：缩减恶意样本数量...")
    mal_df = mal_df.sample(n=ben_count, random_state=42)

# 合并结果
df_balanced = pd.concat([ben_df, mal_df])
print(f"均衡后总样本数: {len(df_balanced)}")

# 3. 划分特征与标签
X = df_balanced.drop(columns=['Label'])
keep_features = [
    'Flow Duration', 'Total Fwd Packet', 'Total Length of Fwd Packet',
    'Total Length of Bwd Packet', 'Fwd Packet Length Max',
    'Fwd Packet Length Min', 'Fwd Packet Length Mean', 'Fwd Packet Length Std',
    'Bwd Packet Length Max', 'Bwd Packet Length Min', 'Bwd Packet Length Mean',
    'Flow Bytes/s', 'Flow Packets/s', 'Flow IAT Mean', 'Flow IAT Std',
    'Flow IAT Max', 'Flow IAT Min', 'Fwd IAT Total', 'Fwd IAT Mean',
    'Fwd IAT Min', 'Bwd IAT Mean', 'Bwd IAT Min', 'Fwd Header Length',
    'Fwd Packets/s', 'Bwd Packets/s', 'Packet Length Min', 'Packet Length Max',
    'Packet Length Mean', 'Packet Length Std', 'Packet Length Variance',
    'PSH Flag Count', 'ACK Flag Count', 'Average Packet Size',
    'Fwd Segment Size Avg', 'Bwd Segment Size Avg', 'Bwd Bytes/Bulk Avg',
    'Subflow Fwd Bytes', 'FWD Init Win Bytes', 'Bwd Init Win Bytes',
    'Fwd Seg Size Min'
]
X_filtered = X[keep_features]
y = df_balanced['Label']
X_train, X_test, y_train, y_test = train_test_split(X_filtered,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=42)

# X 通常是去掉了 'Label' 列后的 DataFrame
print(f"特征总数: {X_filtered.shape[1]}")
print("特征列表清单：")
print(X_filtered.columns.tolist())

# 4. 训练随机森林
print(f"启动随机森林训练 (训练集规模: {X_train.shape})...")
rf = RandomForestClassifier(n_estimators=100,
                            n_jobs=-1,
                            random_state=42,
                            max_depth=None)
rf.fit(X_train, y_train)

# 5. 评估结果
y_pred = rf.predict(X_test)
print("\n--- 训练评估报告 ---")
print(classification_report(y_test, y_pred))

# 6. 交叉验证 (核心补充)
print("\n正在启动交叉验证 (这可能需要一些时间)...")

# 定义 kf：将数据随机打乱(shuffle)，并切分成 5 份
kf = KFold(n_splits=5, shuffle=True, random_state=42)

scoring_metrics = ['accuracy', 'precision', 'recall', 'f1']
results = cross_validate(rf, X_filtered, y, cv=kf, scoring=scoring_metrics)

print("\n--- 5折交叉验证综合评估 ---")
print(f"平均准确率 (Accuracy):  {results['test_accuracy'].mean():.4f}")
print(f"平均精确率 (Precision): {results['test_precision'].mean():.4f}")
print(f"平均召回率 (Recall):    {results['test_recall'].mean():.4f}")
print(f"平均 F1 分数:          {results['test_f1'].mean():.4f}")


def plot_combined_results(results, importances):
    # 1. 准备交叉验证数据
    plot_data = pd.DataFrame({
        'Accuracy': results['test_accuracy'],
        'Precision': results['test_precision'],
        'Recall': results['test_recall'],
        'F1-Score': results['test_f1']
    })

    # 2. 创建画布：1行2列
    # figsize(宽, 高) 可以根据需要调整
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    sns.set(style="whitegrid")

    # --- 左图：交叉验证箱线图 ---
    sns.boxplot(data=plot_data, palette="Set3", width=0.5, ax=ax1)
    sns.stripplot(data=plot_data, color=".25", size=6, jitter=True, ax=ax1)
    ax1.set_title('A. 5-Fold Cross Validation Metrics', fontsize=16, pad=15)
    ax1.set_ylabel('Score', fontsize=13)
    ax1.set_ylim(0, 1.05)

    # --- 右图：特征重要性条形图 ---
    top_10_features = importances.head(10)
    sns.barplot(x=top_10_features.values,
                y=top_10_features.index,
                palette="viridis",
                ax=ax2)
    ax2.set_title('B. Top 10 Important Features', fontsize=16, pad=15)
    ax2.set_xlabel('Importance Score', fontsize=13)
    ax2.set_ylabel('Features', fontsize=13)

    # 3. 自动调整布局，防止标签重叠
    plt.tight_layout()

    # 4. 保存为高清图片
    plt.savefig('model_evaluation_combined.png', dpi=300, bbox_inches='tight')
    print("--- 拼图已保存为 model_evaluation_combined.png ---")

    # 5. 显示
    plt.show()


importances = pd.Series(rf.feature_importances_,
                        index=X_filtered.columns).sort_values(ascending=False)
plot_combined_results(results, importances)

# 1. 找出贡献度小于 0.01 的特征名
# importances = pd.Series(rf.feature_importances_, index=X_filtered.columns)
# low_importance_features = importances[importances < 0.01]
# print(f"贡献度极低的特征数量: {len(low_importance_features)}")
# print(low_importance_features.index.tolist())

# 2. 找出贡献度大于 0.01 (或更高) 的特征名
# keep_features = importances[importances >= 0.01].index.tolist()

# print(f"保留的高权重特征数量: {len(keep_features)}")
# print(keep_features)

# 1. 提取特征重要性并匹配特征名
# 确保 rf 是你训练好的随机森林对象，X_filtered 是你筛选后的特征矩阵
importances = rf.feature_importances_
feature_names = X_filtered.columns

# 2. 创建一个 DataFrame 方便排序和展示
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# 3. 打印出所有特征的重要性数值（控制台查看）
print("\n--- 所有特征重要性排序 ---")
pd.set_option('display.max_rows', None)  # 确保显示所有行
print(feature_importance_df)

# 4. 可视化展示（条形图）
plt.figure(figsize=(10, 12))  # 因为有45个特征，高度设长一点
sns.barplot(x='Importance',
            y='Feature',
            data=feature_importance_df,
            palette='viridis')
plt.title('Feature Importance Ranking (All 45 Features)')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.tight_layout()

# 保存图片
plt.savefig('all_features_importance.png')
plt.show()
