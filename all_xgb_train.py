import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import multiprocessing
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, cross_validate
from sklearn.feature_selection import RFECV
from sklearn.metrics import classification_report


# ==========================================
# 1. 绘图函数：最终模型性能与特征重要性
# ==========================================
def plot_final_evaluation(results, model, features):
    """
    可视化 5-Fold 交叉验证结果与特征重要性排名
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
    ax1.set_title('A. Cross-Validation Performance (XGBoost)', fontsize=14)

    # B. Top 10 核心特征 (条形图)
    sns.barplot(x=importances.head(10).values,
                y=importances.head(10).index,
                palette="viridis",
                ax=ax2)
    ax2.set_title('B. Top 10 Important Features (XGBoost)', fontsize=14)

    plt.tight_layout()
    plt.savefig('final_xgboost_report.png', dpi=300)
    plt.show()


# ==========================================
# 2. 主程序入口
# ==========================================
if __name__ == "__main__":

    # --- 步骤 1: 数据准备 ---
    print("正在加载数据...")
    df = pd.read_parquet("processed_data.parquet")

    # 样本均衡
    mal_df = df[df['Label'] == 1]
    ben_df = df[df['Label'] == 0]
    sample_size = min(len(mal_df), len(ben_df))
    df_balanced = pd.concat([
        mal_df.sample(n=sample_size, random_state=42),
        ben_df.sample(n=sample_size, random_state=42)
    ]).sample(frac=1, random_state=42)

    # --- 步骤 2: 特征工程 ---
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
    X_filtered = X[keep_features].copy()
    X_filtered['Flow_Density'] = X_filtered['Total Length of Fwd Packet'] / (
        X_filtered['Flow Duration'] + 1e-5)
    X_filtered['Packet_Size_Stability'] = X_filtered['Packet Length Std'] / (
        X_filtered['Packet Length Mean'] + 1e-5)

    y = df_balanced['Label']
    X_train, X_test, y_train, y_test = train_test_split(X_filtered,
                                                        y,
                                                        test_size=0.2,
                                                        random_state=42)

    # --- 步骤 3: XGBoost-RFECV 特征筛选 ---
    print("\n--- 步骤 1: 正在进行 XGBoost-RFECV 筛选 (寻找最佳特征子集) ---")
    xgb_selector = XGBClassifier(n_estimators=100,
                                 max_depth=6,
                                 learning_rate=0.1,
                                 tree_method='hist',
                                 random_state=42,
                                 n_jobs=-1,
                                 eval_metric='logloss')

    # cv=3 兼顾速度与准确度
    rfecv = RFECV(estimator=xgb_selector,
                  step=2,
                  cv=3,
                  scoring='f1',
                  n_jobs=-1)
    rfecv.fit(X_train, y_train)

    print(f"最佳特征数量: {rfecv.n_features_}")

    # 【新增：绘制 RFECV 学习曲线】
    plt.figure(figsize=(10, 6))
    plt.plot(range(1,
                   len(rfecv.cv_results_['mean_test_score']) + 1),
             rfecv.cv_results_['mean_test_score'],
             marker='o',
             linestyle='--')
    plt.xlabel("Number of Features Selected (Step=2)")
    plt.ylabel("Cross-Validation F1-Score")
    plt.title("RFECV Learning Curve: Impact of Feature Count on Performance")
    plt.grid(True)
    plt.savefig('rfecv_learning_curve.png', dpi=300)
    plt.show()

    optimal_features = X_train.columns[rfecv.support_].tolist()
    X_train_optimal = X_train[optimal_features]
    X_test_optimal = X_test[optimal_features]

    # --- 步骤 4: 参数调优 (GridSearch) ---
    print("\n--- 步骤 2: 正在进行参数调优 (GridSearch) ---")
    xgb_model = XGBClassifier(tree_method='hist',
                              random_state=42,
                              n_jobs=-1,
                              eval_metric='logloss')

    param_grid_xgb = {
        'n_estimators': [100, 300],
        'learning_rate': [0.05, 0.1],
        'max_depth': [6, 10],
        'scale_pos_weight': [3, 5]
    }

    grid_search = GridSearchCV(xgb_model,
                               param_grid_xgb,
                               cv=3,
                               scoring='f1',
                               verbose=1,
                               n_jobs=-1)
    grid_search.fit(X_train_optimal, y_train)
    best_xgb = grid_search.best_estimator_

    # --- 步骤 5: 最终评估 ---
    print("\n--- 步骤 3: 最终性能验证 ---")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = cross_validate(
        best_xgb,
        X_filtered[optimal_features],
        y,
        cv=kf,
        scoring=['accuracy', 'precision', 'recall', 'f1'])

    print(f"\n【平均性能数据】")
    print(f"召回率 (Recall): {cv_results['test_recall'].mean():.4f}")
    print(f"F1 分数: {cv_results['test_f1'].mean():.4f}")

    y_pred = best_xgb.predict(X_test_optimal)
    print("\n测试集详细分类报告:")
    print(classification_report(y_test, y_pred))

    # 调用绘图
    plot_final_evaluation(cv_results, best_xgb, X_filtered[optimal_features])
