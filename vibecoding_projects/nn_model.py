
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# 中文支持
plt.rcParams['font.sans-serif'] = ['STHeiti', 'PingFang HK', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
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

features = ['THM', 'pH', 'Light', 'CatalystDosage', 'Time']
label = 'DE'
print(f"总样本数: {len(df)}")

# ── 按实验条件分组 ──────────────────────────────────────────────────
# 每组内实验条件不变，只有时间或其他单变量变化
# 根据数据顺序定义 group boundaries
group_boundaries = [0, 5, 10, 16, 22, 27, 32, 36, 39]  # 8个实验组
print(f"实验组数: {len(group_boundaries) - 1}")
for i in range(len(group_boundaries) - 1):
    start, end = group_boundaries[i], group_boundaries[i+1]
    print(f"  组{i+1}: 行{start}-{end-1} (共{end-start}条)")

# ── MinMax 归一化 ───────────────────────────────────────────────────
x_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()
X_scaled = x_scaler.fit_transform(df[features].values)
y_scaled = y_scaler.fit_transform(df[[label]].values)

# ── 构建滑动窗口序列 ──────────────────────────────────────────────
SEQUENCE_LENGTH = 3

X_sequences, y_sequences = [], []

skipped_groups = []
for i in range(len(group_boundaries) - 1):
    start, end = group_boundaries[i], group_boundaries[i+1]
    group_len = end - start

    if group_len < SEQUENCE_LENGTH + 1:
        skipped_groups.append(f"组{i+1}({group_len}条)")
        continue

    # 在组内滑动窗口：用前 SEQUENCE_LENGTH 步预测下一步 DE
    for j in range(group_len - SEQUENCE_LENGTH):
        seq_X = X_scaled[start + j : start + j + SEQUENCE_LENGTH]  # (seq_len, 5)
        seq_y = y_scaled[start + j + SEQUENCE_LENGTH]              # 下一个时间步的 DE

        X_sequences.append(seq_X)
        y_sequences.append(seq_y[0])

X_sequences = np.array(X_sequences)
y_sequences = np.array(y_sequences)

print(f"\n序列形状: {X_sequences.shape}")
print(f"标签形状: {y_sequences.shape}")
# 输出应为 (N, seq_len, 5) — N个序列，每个seq_len时间步，每步5特征
if skipped_groups:
    print(f"跳过的组 (不足{SEQUENCE_LENGTH+1}条): {', '.join(skipped_groups)}")

# ── 划分训练/测试集 (小数据集，按组划分避免数据泄露) ──────────────
# 按组划分：用前6组的序列做训练，后2组的序列做测试
seq_to_group = []
for i in range(len(group_boundaries) - 1):
    start, end = group_boundaries[i], group_boundaries[i+1]
    group_len = end - start
    if group_len >= SEQUENCE_LENGTH + 1:
        n_seqs = group_len - SEQUENCE_LENGTH
        seq_to_group.extend([i] * n_seqs)
seq_to_group = np.array(seq_to_group)

# 按实验轮次划分：前70%组训练，后30%测试
groups = np.unique(seq_to_group)
n_train_groups = max(1, int(len(groups) * 0.7))
train_groups = groups[:n_train_groups]
test_groups = groups[n_train_groups:]

train_mask = np.isin(seq_to_group, train_groups)
test_mask = np.isin(seq_to_group, test_groups)

X_train = X_sequences[train_mask]
X_test = X_sequences[test_mask]
y_train = y_sequences[train_mask]
y_test = y_sequences[test_mask]

print(f"训练组: {train_groups.tolist()}, 训练集: {X_train.shape[0]} 序列")
print(f"测试组: {test_groups.tolist()}, 测试集: {X_test.shape[0]} 序列")

# ── 转 Tensor ──────────────────────────────────────────────────────
X_train_t = torch.tensor(X_train, dtype=torch.float32)
X_test_t  = torch.tensor(X_test, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
y_test_t  = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# ── Dataset / DataLoader ───────────────────────────────────────────
class PhotocatalysisDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = PhotocatalysisDataset(X_train_t, y_train_t)
test_dataset  = PhotocatalysisDataset(X_test_t, y_test_t)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=2, shuffle=False)

# ── 模型定义 ───────────────────────────────────────────────────────
class AttentionBlock(nn.Module):
    """特征级别的自注意力"""
    def __init__(self, feature_dim):
        super().__init__()
        self.attention = nn.Linear(feature_dim, feature_dim)
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x):
        # x: (batch, seq_len, features)
        weights = self.softmax(self.attention(x))  # (batch, seq_len, features)
        return x * weights

class GRN(nn.Module):
    """Gated Residual Network"""
    def __init__(self, dim):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)
        self.elu = nn.ELU()
        self.gate = nn.Linear(dim, dim)
        self.sigmoid = nn.Sigmoid()
        self.layer_norm = nn.LayerNorm(dim)
    def forward(self, x):
        residual = x
        out = self.elu(self.linear1(x))
        out = self.linear2(out)
        gate_weight = self.sigmoid(self.gate(x))
        out = gate_weight * out
        return self.layer_norm(residual + out)

class PhotocatalysisLSTM(nn.Module):
    """LSTM + Attention + GRN 混合模型"""
    def __init__(self, input_dim=5):
        super().__init__()
        self.attention = AttentionBlock(input_dim)
        self.grn = GRN(input_dim)
        self.lstm1 = nn.LSTM(input_size=input_dim, hidden_size=32, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=32, hidden_size=16, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(16, 16)
        self.fc2 = nn.Linear(16, 1)
        self.elu = nn.ELU()

    def forward(self, x):
        x = self.attention(x)   # 特征注意力
        x = self.grn(x)         # 门控残差
        x, _ = self.lstm1(x)    # 第一层 LSTM
        x, _ = self.lstm2(x)    # 第二层 LSTM
        x = x[:, -1, :]         # 取最后时间步
        x = self.dropout(x)
        x = self.elu(self.fc1(x))
        output = self.fc2(x)
        return output

# ── 训练 ───────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

model = PhotocatalysisLSTM().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

EPOCHS = 300
train_losses, test_losses = [], []

for epoch in range(EPOCHS):
    # 训练
    model.train()
    epoch_loss = 0.0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        pred = model(X_batch)
        loss = criterion(pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    avg_train_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # 测试
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            test_loss += loss.item()
    avg_test_loss = test_loss / len(test_loader)
    test_losses.append(avg_test_loss)

    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch+1:3d}/{EPOCHS}  "
              f"Train Loss: {avg_train_loss:.6f}  Test Loss: {avg_test_loss:.6f}")

# ── 评估 ───────────────────────────────────────────────────────────
model.eval()
with torch.no_grad():
    pred_scaled = model(X_test_t.to(device)).cpu().numpy()
    actual_scaled = y_test_t.numpy()

# 反归一化
pred_de = y_scaler.inverse_transform(pred_scaled)
actual_de = y_scaler.inverse_transform(actual_scaled)

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
r2 = r2_score(actual_de, pred_de)
rmse = np.sqrt(mean_squared_error(actual_de, pred_de))
mae = mean_absolute_error(actual_de, pred_de)

print(f"\n{'='*50}")
print(f"测试集评估结果:")
print(f"  R²   = {r2:.4f}")
print(f"  RMSE = {rmse:.3f}")
print(f"  MAE  = {mae:.3f}")
print(f"{'='*50}")

print("\n测试集详细对比:")
for i in range(len(actual_de)):
    print(f"  实际 DE={actual_de[i][0]:.1f}  →  预测 DE={pred_de[i][0]:.1f}  "
          f"(Δ={actual_de[i][0]-pred_de[i][0]:+.1f})")

# ── 可视化 ───────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# 1. 训练曲线
ax = axes[0]
ax.plot(train_losses, label='Train', linewidth=1.5, color='#ff6b6b')
ax.plot(test_losses, label='Test', linewidth=1.5, color='#4ecdc4')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss (MSE)')
ax.set_title('训练曲线')
ax.legend()
ax.grid(alpha=0.3)

# 2. 实际 vs 预测
ax = axes[1]
min_val = min(actual_de.min(), pred_de.min()) - 5
max_val = max(actual_de.max(), pred_de.max()) + 5
ax.scatter(actual_de, pred_de, alpha=0.7, s=50, c='#45b7d1', edgecolors='white', linewidth=0.5)
ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.3)
ax.set_xlabel('实际 DE')
ax.set_ylabel('预测 DE')
ax.set_title(f'实际 vs 预测 (R²={r2:.3f})')
ax.axis('equal')

# 3. 残差
ax = axes[2]
residuals = actual_de - pred_de
ax.scatter(pred_de, residuals, alpha=0.7, s=50, c='#ff6b6b', edgecolors='white', linewidth=0.5)
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('预测 DE')
ax.set_ylabel('残差 (实际 - 预测)')
ax.set_title('残差分布')
ax.grid(alpha=0.3)

plt.tight_layout()
plot_path = '/Users/wangdajin/电脑内文稿/cursor/pomodoro/nn_eval.png'
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"\n评估图已保存: {plot_path}")

# ── 保存模型 ───────────────────────────────────────────────────────
torch.save({
    'model_state_dict': model.state_dict(),
    'x_scaler': x_scaler,
    'y_scaler': y_scaler,
    'input_dim': 5,
    'seq_len': SEQUENCE_LENGTH,
}, '/Users/wangdajin/电脑内文稿/cursor/pomodoro/nn_model.pth')
print("模型已保存: nn_model.pth")
