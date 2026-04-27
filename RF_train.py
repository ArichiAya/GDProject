import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_validate, KFold, GridSearchCV
from sklearn.feature_selection import RFECV
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import multiprocessing


# ==========================================
# 绘图函数定义 (放在入口外面)
# ==========================================
def plot_results(results, model, features):
    """
    可视化交叉验证结果与特征重要性
    """
    importances = pd.Series(
        model.feature_importances_,
        index=features.columns).sort_values(ascending=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    # A. 交叉验证结果 (箱线图)
    cv_df = pd.DataFrame({
        'Accuracy': results['test_accuracy'],
        'Precision': results['test_precision'],
        'Recall': results['test_recall'],
        'F1-Score': results['test_f1']
    })
    sns.boxplot(data=cv_df, palette="Set3", ax=ax1)
    ax1.set_title('A. Cross-Validation Reliability (Optimal Features)',
                  fontsize=14)

    # B. Top 10 核心特征 (条形图)
    sns.barplot(x=importances.head(10).values,
                y=importances.head(10).index,
                palette="viridis",
                ax=ax2)
    ax2.set_title('B. Top 10 Important Features (After RFE)', fontsize=14)

    plt.tight_layout()
    plt.savefig('evaluation_report_optimal.png', dpi=300)
    plt.show()

    # 打印全量排名
    print("\n--- 调优后全量特征重要性排名 ---")
    fi_df = pd.DataFrame({
        'Feature': importances.index,
        'Importance': importances.values
    })
    print(fi_df)


# ==========================================
# 主程序入口 (Windows 多进程必须包裹在此处)
# ==========================================
if __name__ == "__main__":

    # 可选：如果依然报错，取消下面这一行的注释，指定一个纯英文的临时路径
    # os.environ['JOBLIB_TEMP_FOLDER'] = 'C:/tmp'

    # 1. 加载数据
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
    ]).sample(frac=1, random_state=42)
    print(f"均衡后总样本数: {len(df_balanced)}")

    # 3. 特征初步筛选与工程
    X = df_balanced.drop(columns=['Label'])
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

    # 使用 .copy() 避免赋值警告
    X_filtered = X[keep_features].copy()

    # 特征增强：流量密度与稳定性指标
    X_filtered['Flow_Density'] = X_filtered['Total Length of Fwd Packet'] / (
        X_filtered['Flow Duration'] + 1e-5)
    X_filtered['Packet_Size_Stability'] = X_filtered['Packet Length Std'] / (
        X_filtered['Packet Length Mean'] + 1e-5)

    print(f"特征工程完成，当前初始特征总数: {X_filtered.shape[1]}")

    y = df_balanced['Label']
    X_train, X_test, y_train, y_test = train_test_split(X_filtered,
                                                        y,
                                                        test_size=0.2,
                                                        random_state=42)

    # 4. RFECV 自动化特征消除 (寻找最优特征数量)
    print("\n--- 步骤 1: 正在通过 RFECV 寻找最佳特征数量 ---")
    n_cpus = multiprocessing.cpu_count()
    rf_rfecv = RandomForestClassifier(n_estimators=100,
                                      max_depth=25,
                                      n_jobs=n_cpus - 1,
                                      random_state=42)

    # 5折交叉验证，以 F1 分数为优化目标
    rfecv = RFECV(estimator=rf_rfecv,
                  step=1,
                  cv=5,
                  scoring='f1',
                  n_jobs=n_cpus - 1)
    rfecv.fit(X_train, y_train)

    print(f"最佳特征数量: {rfecv.n_features_}")
    optimal_features = X_train.columns[rfecv.support_].tolist()
    print(f"选出的最优特征清单: {optimal_features}")

    # 绘制 RFECV 学习曲线
    plt.figure(figsize=(10, 5))
    plt.plot(range(1,
                   len(rfecv.cv_results_['mean_test_score']) + 1),
             rfecv.cv_results_['mean_test_score'])
    plt.xlabel("Number of Features Selected")
    plt.ylabel("CV F1-Score")
    plt.title("RFECV: Optimization of Feature Count")
    plt.grid(True)
    plt.savefig('rfecv_curve.png')
    plt.show()

    # 应用最优特征集
    X_train_optimal = X_train[optimal_features]
    X_test_optimal = X_test[optimal_features]

    # 5. GridSearchCV 自动化调优 (在最优特征集上寻找最佳参数)
    print("\n--- 步骤 2: 正在最优特征集上进行参数调优 ---")
    rf_base = RandomForestClassifier(random_state=42, n_jobs=n_cpus - 1)
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 15, 25],
        'min_samples_split': [2, 5]
    }

    grid_search = GridSearchCV(estimator=rf_base,
                               param_grid=param_grid,
                               cv=3,
                               scoring='f1',
                               verbose=2,
                               n_jobs=n_cpus - 1)
    grid_search.fit(X_train_optimal, y_train)

    print(f"\n最佳参数组合: {grid_search.best_params_}")
    best_rf = grid_search.best_estimator_

    # 6. 最终评估 (测试集)
    y_pred = best_rf.predict(X_test_optimal)
    print("\n--- 调优与特征筛选后最终评估报告 ---")
    print(classification_report(y_test, y_pred))

    # 7. 全局稳定性校验 (5折交叉验证)
    print("\n正在启动最终 5-Fold 交叉验证...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = cross_validate(
        best_rf,
        X_filtered[optimal_features],
        y,
        cv=kf,
        scoring=['accuracy', 'precision', 'recall', 'f1'])

    print(f"平均准确率: {cv_results['test_accuracy'].mean():.4f}")
    print(f"平均精确率: {cv_results['test_precision'].mean():.4f}")
    print(f"平均召回率: {cv_results['test_recall'].mean():.4f}")
    print(f"平均 F1 分数: {cv_results['test_f1'].mean():.4f}")

    # 8. 调用绘图展示
    plot_results(cv_results, best_rf, X_filtered[optimal_features])
