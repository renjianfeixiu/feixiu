import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


columns = [
    'THM',
    'pH',
    'Light',
    'CatalystDosage',
    'Time',
    'DE'
]

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
features = [
    'THM',
    'pH',
    'Light',
    'CatalystDosage',
    'Time'
]
label = 'DE'
X = df[features].values
y = df[label].values.reshape(-1, 1)
#minmax归一化
x_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()

X_scaled = x_scaler.fit_transform(X)
y_scaled = y_scaler.fit_transform(y)
sequence_length = 5
X_sequences = []
y_sequences = []

# 20条一个组
group_size = 20

for start in range(0, len(X_scaled), group_size):

    # 取一个实验组
    X_group = X_scaled[start:start + group_size]
    y_group = y_scaled[start:start + group_size]


X_sequences = np.array(X_sequences)
y_sequences = np.array(y_sequences)

print("Sequence shape:", X_sequences.shape)
print("Label shape:", y_sequences.shape)
#转tensor
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

class PhotocatalysisDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = PhotocatalysisDataset(X_train, y_train)
train_loader = DataLoader(
    train_dataset,
    batch_size=2,
    shuffle=False
)
class attention(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.attention = nn.Linear(feature_dim, feature_dim)
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x):
        weights = self.softmax(self.attention(x))
        weighted = x * weights
        return weighted
class GRN(nn.Module):
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
    def __init__(self):
        super().__init__()
        # Attention
        self.attention = AttentionBlock(5)
        # GRN
        self.grn = GRN(5)
        # 第一层LSTM
        self.lstm1 = nn.LSTM(
            input_size=5,
            hidden_size=32,
            batch_first=True)
        # 第二层LSTM
        self.lstm2 = nn.LSTM(
            input_size=32,
            hidden_size=16,
            batch_first=True)
        self.dropout = nn.Dropout(0.2)
        #全连接
        self.fc1 = nn.Linear(16, 16)
        self.fc2 = nn.Linear(16, 1)
        self.elu = nn.ELU()

    def forward(self, x):
        #a
        x = self.attention(x)
        #g
        x = self.grn(x)
        #l
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)

        x = x[:, -1, :]
        #dropout
        x = self.dropout(x)
        # 全连接
        x = self.elu(self.fc1(x))
        output = self.fc2(x)
        return output

model = PhotocatalysisLSTM()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.003)
train_losses = []
epochs = 200
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for X_batch, y_batch in train_loader:
        # 前向传播
        pred = model(X_batch)
        loss = criterion(pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    avg_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_loss)
model.eval()

with torch.no_grad():
    pred_scaled = model(X_test).numpy()
plt.plot(train_losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Curve')
plt.show()
