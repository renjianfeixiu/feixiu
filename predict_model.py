#!/opt/anaconda3/bin/python3
"""
催化降解效率预测模型训练
基于实验数据（THM, pH, Light, CatalystDosage, Time）预测 DE
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ── 数据 ──────────────────────────────────────────────────────────────
columns = ['THM', 'pH', 'Light', 'CatalystDosage', 'Time', 'DE']
data = [
    [5.12, 5.67, 520, 25, 10, 15.1],
    [5.12, 5.67, 520, 25, 20, 28.4],
    [5.12, 5.67, 520, 25, 30, 41.5],
    [5.12, 5.67, 520, 25, 40, 52.7],
    [5.12, 5.67, 520, 25, 50, 61.1],
    [19.42, 7.20, 165, 25, 10, 5.1],
    [19.42, 7.20, 165, 25, 20, 8.2],
    [19.42, 7.20, 165, 25, 30, 11.3],
    [19.42, 7.20, 165, 25, 40, 14.5],
    [19.42, 7.20, 165, 25, 50, 17.9],
    [5.00, 5.70, 520, 1.00, 30, 35.0],
    [5.00, 5.70, 520, 1.00, 60, 47.0],
    [5.00, 5.70, 520, 1.00, 90, 53.0],
    [5.00, 5.70, 520, 1.00, 120, 59.0],
    [5.00, 5.70, 520, 1.00, 150, 63.0],
    [5.00, 5.70, 520, 1.00, 180, 65.7],
    [5.00, 5.70, 520, 1.00, 30, 42.0],
    [5.00, 5.70, 520, 1.00, 60, 55.0],
    [5.00, 5.70, 520, 1.00, 90, 66.0],
    [5.00, 5.70, 520, 1.00, 120, 75.0],
    [5.00, 5.70, 520, 1.00, 150, 81.0],
    [5.00, 5.70, 520, 1.00, 180, 84.6],
    [5.00, 3.00, 520, 1.00, 180, 58.2],
    [5.00, 5.00, 520, 1.00, 180, 79.3],
    [5.00, 7.00, 520, 1.00, 180, 72.1],
    [5.00, 9.00, 520, 1.00, 180, 67.5],
    [5.00, 11.00, 520, 1.00, 180, 51.4],
    [5.00, 5.70, 520, 1.00, 180, 84.6],
    [10.0, 5.70, 520, 1.00, 180, 72.3],
    [20.0, 5.70, 520, 1.00, 180, 57.8],
    [30.0, 5.70, 520, 1.00, 180, 44.2],
    [50.0, 5.70, 520, 1.00, 180, 29.5],
    [5.00, 5.70, 520, 0.5, 180, 61.3],
    [5.00, 5.70, 520, 1.0, 180, 84.6],
    [5.00, 5.70, 520, 1.5, 180, 81.2],
    [5.00, 5.70, 520, 2.0, 180, 74.9],
    [5.00, 5.70, 165, 1.00, 180, 47.6],
    [5.00, 5.70, 331, 1.00, 180, 68.9],
    [5.00, 5.70, 520, 1.00, 180, 84.6],
]
df = pd.DataFrame(data, columns=columns)
print(f"样本数: {len(df)}")
print(f"特征: {columns}")

# ── 特征工程 ─────────────────────────────────────────────────────────
# 原始特征
base_features = ['THM', 'pH', 'Light', 'CatalystDosage', 'Time']

# 添加非线性变换
df['pH_sq'] = df['pH'] ** 2                # pH 最优区间效应
df['LogTime'] = np.log1p(df['Time'])        # 时间饱和趋势
df['THM_inv'] = 1 / (df['THM'] + 0.01)      # 底物浓度倒数效应

engineered_features = base_features + ['pH_sq', 'LogTime', 'THM_inv']
feature_names = engineered_features

X = df[engineered_features].values
y = df['DE'].values

print(f"特征工程后维度: {X.shape[1]} 个特征")

# ── 模型与评估 ──────────────────────────────────────────────────────
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut, cross_val_score, cross_val_predict, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# 对比模型
models = {}

# 1. Random Forest
models['Random Forest'] = RandomForestRegressor(
    n_estimators=500, max_depth=6, min_samples_leaf=2,
    random_state=42
)

# 2. XGBoost (小数据强正则化)
try:
    from xgboost import XGBRegressor
    models['XGBoost'] = XGBRegressor(
        n_estimators=200, max_depth=3, learning_rate=0.08,
        reg_lambda=5.0, reg_alpha=2.0,
        subsample=0.7, colsample_bytree=0.8,
        random_state=42
    )
except ImportError:
    print("XGBoost 未安装，跳过")

# 3. Ridge + PolynomialFeatures (degree=2)
models['Ridge+Poly2'] = Pipeline([
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('scaler', StandardScaler()),
    ('ridge', Ridge(alpha=10.0, random_state=42))
])

# ── LOOCV 评估 ──────────────────────────────────────────────────────
loo = LeaveOneOut()
results = {}

print("\n" + "="*60)
print("留一法交叉验证 (LOOCV) 结果")
print("="*60)

for name, model in models.items():
    # 需要手动 LOOCV 因为不同的模型需要不同的预处理
    y_true_cv, y_pred_cv = [], []
    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # 标准化（PolynomialFeatures pipeline 自带）
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        if name == 'Ridge+Poly2':
            # Pipeline 自己处理 poly + scaling + ridge
            pipe = Pipeline([
                ('poly', PolynomialFeatures(degree=2, include_bias=False)),
                ('scaler', StandardScaler()),
                ('ridge', Ridge(alpha=10.0, random_state=42))
            ])
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)
        else:
            model_clone = model.__class__(**model.get_params())
            model_clone.fit(X_train_scaled, y_train)
            y_pred = model_clone.predict(X_test_scaled)

        y_true_cv.append(y_test[0])
        y_pred_cv.append(y_pred[0])

    r2 = r2_score(y_true_cv, y_pred_cv)
    rmse = np.sqrt(mean_squared_error(y_true_cv, y_pred_cv))
    mae = mean_absolute_error(y_true_cv, y_pred_cv)

    results[name] = {
        'y_true': np.array(y_true_cv),
        'y_pred': np.array(y_pred_cv),
        'r2': r2,
        'rmse': rmse,
        'mae': mae,
    }

    print(f"\n{'─'*40}")
    print(f"模型: {name}")
    print(f"  R²  = {r2:.4f}")
    print(f"  RMSE= {rmse:.3f}")
    print(f"  MAE = {mae:.3f}")

# 找出最佳模型
best_model_name = max(results, key=lambda k: results[k]['r2'])
print(f"\n{'='*60}")
print(f"最佳模型: {best_model_name}  (R² = {results[best_model_name]['r2']:.4f})")

# ── 最终模型训练（全量数据） ──────────────────────────────────────
print(f"\n训练最终模型: {best_model_name} ...")

final_scaler = StandardScaler()
X_scaled = final_scaler.fit_transform(X)

if best_model_name == 'Ridge+Poly2':
    final_model = Pipeline([
        ('poly', PolynomialFeatures(degree=2, include_bias=False)),
        ('scaler', StandardScaler()),
        ('ridge', Ridge(alpha=10.0, random_state=42))
    ])
    final_model.fit(X, y)
else:
    final_model = models[best_model_name].__class__(**models[best_model_name].get_params())
    final_model.fit(X_scaled, y)

# ── 保存模型 ─────────────────────────────────────────────────────────
import pickle
import os

model_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(model_dir, 'best_model.pkl')

with open(model_path, 'wb') as f:
    pickle.dump({
        'model': final_model,
        'scaler': final_scaler,
        'model_name': best_model_name,
        'feature_names': feature_names,
        'X_mean': X.mean(axis=0),
        'X_std': X.std(axis=0),
    }, f)

print(f"模型已保存: {model_path}")

# ── 可视化 ───────────────────────────────────────────────────────────
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# 中文支持
plt.rcParams['font.sans-serif'] = ['STHeiti', 'PingFang HK', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

fig, axes = plt.subplots(2, 2, figsize=(14, 11))

# 1. 实际 vs 预测 (LOOCV)
ax = axes[0, 0]
for name in results:
    ax.scatter(results[name]['y_true'], results[name]['y_pred'],
               alpha=0.6, s=30, label=name)
min_val, max_val = y.min() - 5, y.max() + 5
ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.3)
ax.set_xlabel('实际 DE')
ax.set_ylabel('预测 DE')
ax.set_title('LOOCV: 实际 vs 预测')
ax.legend(fontsize=8)
ax.axis('equal')

# 2. 最佳模型详细对比
ax = axes[0, 1]
best = results[best_model_name]
residuals = best['y_true'] - best['y_pred']
ax.scatter(best['y_pred'], residuals, alpha=0.6, s=40, c='#ff6b6b')
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('预测 DE')
ax.set_ylabel('残差 (实际 - 预测)')
ax.set_title(f'{best_model_name} 残差分布')
# 标注最大残差点
max_resid_idx = np.argmax(np.abs(residuals))
ax.annotate(f'{residuals[max_resid_idx]:.1f}',
            (best['y_pred'][max_resid_idx], residuals[max_resid_idx]),
            xytext=(5, 5), textcoords='offset points', fontsize=8)

# 3. 特征重要性（仅对 RF/XGBoost 有效）
ax = axes[1, 0]
if hasattr(final_model, 'feature_importances_'):
    importances = final_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    bars = ax.barh(range(len(indices)), importances[indices][::-1], color='#4ecdc4')
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices[::-1]], fontsize=9)
    ax.set_xlabel('重要性')
    ax.set_title('特征重要性')
else:
    # Ridge+Poly2: 用系数绝对值代替
    if best_model_name == 'Ridge+Poly2':
        coefs = final_model.named_steps['ridge'].coef_
        poly = final_model.named_steps['poly']
        poly_feature_names = poly.get_feature_names_out(feature_names)
        abs_coefs = np.abs(coefs)
        top_n = min(10, len(abs_coefs))
        top_idx = np.argsort(abs_coefs)[-top_n:]
        bars = ax.barh(range(top_n), abs_coefs[top_idx], color='#45b7d1')
        ax.set_yticks(range(top_n))
        ax.set_yticklabels([poly_feature_names[i] for i in top_idx], fontsize=8)
        ax.set_xlabel('|系数|')
        ax.set_title(f'Ridge+Poly2  Top{top_n} 特征')
    else:
        ax.text(0.5, 0.5, '该模型无特征重要性', ha='center', va='center', fontsize=12)
        ax.set_title('特征重要性')

# 4. 特征重要性对比 (RF vs XGBoost)
ax = axes[1, 1]
if 'Random Forest' in models and 'XGBoost' in models:
    # 计算 RF 和 XGB 的 feature importance
    rf_imp = {}
    xgb_imp = {}

    if hasattr(models['Random Forest'], 'feature_importances_'):
        models['Random Forest'].fit(X_scaled, y)
        rf_imp = dict(zip(feature_names, models['Random Forest'].feature_importances_))

    if 'XGBoost' in models and hasattr(models['XGBoost'], 'feature_importances_'):
        models['XGBoost'].fit(X_scaled, y)
        xgb_imp = dict(zip(feature_names, models['XGBoost'].feature_importances_))

    if rf_imp and xgb_imp:
        feat_df = pd.DataFrame({'RF': rf_imp, 'XGB': xgb_imp})
        feat_df.plot(kind='bar', ax=ax, color=['#ff6b6b', '#45b7d1'], alpha=0.8)
        ax.set_title('特征重要性对比: RF vs XGBoost')
        ax.set_ylabel('重要性')
        ax.legend(fontsize=8)
        ax.tick_params(axis='x', rotation=45)
    else:
        ax.text(0.5, 0.5, '特征重要性数据不足', ha='center', va='center', fontsize=12)
else:
    ax.text(0.5, 0.5, '仅有一个模型可比较', ha='center', va='center', fontsize=12)
    ax.set_title('特征重要性对比')

plt.tight_layout()
plot_path_img = os.path.join(model_dir, 'model_evaluation.png')
plt.savefig(plot_path_img, dpi=150, bbox_inches='tight')
print(f"评估图已保存: {plot_path_img}")

# ── 部分依赖图 (PDP) ──────────────────────────────────────────────
from sklearn.inspection import PartialDependenceDisplay

fig2, axes2 = plt.subplots(2, 3, figsize=(15, 9))
features_plot = [0, 1, 2, 3, 4]

# 为每个特征做 PDP
if best_model_name == 'Ridge+Poly2':
    # Pipeline 模型的 PDP 需要自行计算
    for idx, f_idx in enumerate(features_plot):
        ax = axes2[idx // 3][idx % 3]
        values = np.linspace(X[:, f_idx].min(), X[:, f_idx].max(), 50)
        preds = []
        for v in values:
            X_copy = X.copy()
            X_copy[:, f_idx] = v
            preds.append(final_model.predict(X_copy).mean())
        ax.plot(values, preds, linewidth=2, color='#ff6b6b')
        ax.scatter(X[:, f_idx], y, alpha=0.4, s=20, color='#8888aa')
        ax.set_xlabel(feature_names[f_idx])
        ax.set_ylabel('预测 DE')
        ax.set_title(f'PDP: {feature_names[f_idx]}')
else:
    PartialDependenceDisplay.from_estimator(
        final_model, X_scaled, features_plot,
        feature_names=feature_names,
        ax=axes2,
        kind='average',
        random_state=42,
    )
    for idx, ax in enumerate(axes2.flat):
        if idx < len(features_plot):
            # 叠加原始数据
            f_idx = features_plot[idx if idx < 5 else 0]
            ax.scatter(X[:, f_idx], y, alpha=0.3, s=20, color='#8888aa')

plt.suptitle('部分依赖图 (PDP) — 各特征对 DE 的边际效应', fontsize=14, y=1.01)
plt.tight_layout()
pdp_path = os.path.join(model_dir, 'partial_dependence.png')
plt.savefig(pdp_path, dpi=150, bbox_inches='tight')
print(f"PDP 图已保存: {pdp_path}")

# ── 打印部分关键样本的预测对比 ─────────────────────────────────────
print(f"\n{'='*60}")
print("部分样本 LOOCV 预测对比:")
print(f"{'No':>3} {'实际DE':>7} {'预测DE':>7} {'误差':>7} {'条件'}")
print('─'*60)

best_pred = results[best_model_name]['y_pred']
best_true = results[best_model_name]['y_true']

# 找出误差最大和最小的几个样本
errors = np.abs(best_true - best_pred)
idx_sorted = np.argsort(errors)

print("\n▶ 预测最佳 (误差最小):")
for i in idx_sorted[:5]:
    err = best_true[i] - best_pred[i]
    row = df.iloc[i]
    print(f"  DE={row['DE']:.1f} → 预测={best_pred[i]:.1f}  "
          f"Δ={err:+.1f}  (THM={row['THM']}, pH={row['pH']}, "
          f"Light={row['Light']}, Cat={row['CatalystDosage']}, t={row['Time']})")

print("\n▶ 预测最差 (误差最大):")
for i in idx_sorted[-5:]:
    err = best_true[i] - best_pred[i]
    row = df.iloc[i]
    print(f"  DE={row['DE']:.1f} → 预测={best_pred[i]:.1f}  "
          f"Δ={err:+.1f}  (THM={row['THM']}, pH={row['pH']}, "
          f"Light={row['Light']}, Cat={row['CatalystDosage']}, t={row['Time']})")

print(f"\n{'='*60}")
print("训练完成！")
