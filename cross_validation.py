import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_validate, KFold, GridSearchCV
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 快速加载数据
print("正在加载数据...")
df = pd.read_parquet("processed_data.parquet")

# 2. 样本均衡处理 (下采样)
mal_df = df[df['Label'] == 1]
ben_df = df[df['Label'] == 0]
sample_size = min(len(mal_df), len(ben_df))

print(f"均衡前：恶意 {len(mal_df)}，正常 {len(ben_df)}")
df_balanced = pd.concat([
    mal_df.sample(n=sample_size, random_state=42),
    ben_df.sample(n=sample_size, random_state=42)
]).sample(frac=1, random_state=42)  # 合并后再次打乱
print(f"均衡后总样本数: {len(df_balanced)}")

# 3. 特征工程
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

# 【修正】使用 .copy() 避免 SettingWithCopyWarning
X_filtered = X[keep_features].copy()

# 【第一步：特征增强】
X_filtered['Flow_Density'] = X_filtered['Total Length of Fwd Packet'] / (
    X_filtered['Flow Duration'] + 1e-5)
X_filtered['Packet_Size_Stability'] = X_filtered['Packet Length Std'] / (
    X_filtered['Packet Length Mean'] + 1e-5)

print(f"特征工程完成，当前特征总数: {X_filtered.shape[1]}")

y = df_balanced['Label']
X_train, X_test, y_train, y_test = train_test_split(X_filtered,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=42)

# 4. 【第二步：自动化调优】
print(f"启动 GridSearch 调优 (训练集规模: {X_train.shape})...")
rf_base = RandomForestClassifier(random_state=42, n_jobs=-1)

param_grid = {
    'n_estimators': [100, 200],  # 缩减了一下范围以加快速度，可自行加回 300
    'max_depth': [15, 25, None],
    'min_samples_split': [2, 5],
    'max_features': ['sqrt']
}

grid_search = GridSearchCV(estimator=rf_base,
                           param_grid=param_grid,
                           cv=3,
                           scoring='f1',
                           verbose=2)
grid_search.fit(X_train, y_train)

print("\n" + "=" * 30)
print(f"最佳参数: {grid_search.best_params_}")
print(f"最佳 F1 分数: {grid_search.best_score_:.4f}")
print("=" * 30)

# 获取最优模型
best_rf = grid_search.best_estimator_

# 5. 评估结果
y_pred = best_rf.predict(X_test)
print("\n--- 最终模型测试集评估报告 ---")
print(classification_report(y_test, y_pred))

# 6. 5折交叉验证 (使用最优模型评估全局稳定性)
print("\n正在启动 5-Fold 交叉验证...")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scoring_metrics = ['accuracy', 'precision', 'recall', 'f1']
cv_results = cross_validate(best_rf,
                            X_filtered,
                            y,
                            cv=kf,
                            scoring=scoring_metrics)

print(f"平均准确率: {cv_results['test_accuracy'].mean():.4f}")
print(f"平均召回率: {cv_results['test_recall'].mean():.4f}")


# 7. 可视化与特征重要性分析
def plot_results(results, model, features):
    importances = pd.Series(
        model.feature_importances_,
        index=features.columns).sort_values(ascending=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    # A. 交叉验证结果
    cv_df = pd.DataFrame({
        'Accuracy': results['test_accuracy'],
        'Precision': results['test_precision'],
        'Recall': results['test_recall'],
        'F1-Score': results['test_f1']
    })
    sns.boxplot(data=cv_df, palette="Set3", ax=ax1)
    ax1.set_title('A. Cross-Validation Reliability')

    # B. Top 10 特征
    sns.barplot(x=importances.head(10).values,
                y=importances.head(10).index,
                palette="viridis",
                ax=ax2)
    ax2.set_title('B. Top 10 Important Features')

    plt.tight_layout()
    plt.savefig('evaluation_report.png', dpi=300)
    plt.show()

    # 打印全量特征重要性排名
    print("\n--- 全量特征重要性排名 ---")
    fi_df = pd.DataFrame({
        'Feature': importances.index,
        'Importance': importances.values
    })
    print(fi_df)


plot_results(cv_results, best_rf, X_filtered)
