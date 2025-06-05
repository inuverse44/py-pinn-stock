import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim # Adamのため
from torch import Tensor # 型アノテーションのため
from torch.utils.data import TensorDataset, DataLoader # 
from typing import List

# データ取得
# ^N255: 日経平均株価のティッカー
nikkei = yf.download("^N225", start="2010-01-01", end="2024-09-30")
print(nikkei)

# データ分割
train_data = nikkei.loc['2010-01-01':'2022-12-31']
test_data = nikkei.loc['2023-01-01':'2024-09-30']

# グラフ（終値）
plt.plot(train_data.index, train_data["Close"], lw=0.5, c='red')
plt.plot(train_data.index, train_data["High"], lw=0.5, c='blue')
plt.grid()
plt.show()
print(f"train_data['Close']: {train_data['Close'].values}")

# データの正規化（0~1）に圧縮
scaler: MinMaxScaler = MinMaxScaler(feature_range=(0, 1))
scaled_data: np.ndarray = scaler.fit_transform(train_data[["Close"]].values)

# 時系列データセット作成
def create_dataset(data: np.ndarray, 
                   time_step: int) -> TensorDataset:
    X, y = [], [] # 入力データと教師データ
    for i in range(time_step, len(data)):
        X.append(data[i-time_step:i, 0])    # 60日間
        y.append(data[i, 0])                # 翌日の値

    X_tensor: Tensor = torch.tensor(np.array(X), dtype=torch.float32)
    y_tensor: Tensor = torch.tensor(np.array(y), dtype=torch.float32)
    dataset: TensorDataset = TensorDataset(X_tensor, y_tensor)
    return dataset

# データ分割 (85%:15%)
time_step: int = 60
train_size: int = int(len(scaled_data) * 0.85)
test_size: int = len(scaled_data) - train_size

train_data: np.ndarray = scaled_data[:train_size]
test_data: np.ndarray = scaled_data[train_size - time_step:] # time_stepぶん重ねて使う（前のtime_stepが必要になるため）

# データセット作成
train_dataset: TensorDataset = create_dataset(train_data, time_step)
test_dataset: TensorDataset = create_dataset(test_data, time_step)

# データローダー作成
batch_size: int = 32
train_loader: DataLoader = DataLoader(train_dataset, batch_size=batch_size)
test_loader: DataLoader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)



# === LSTMモデル定義 ===
class LSTM(nn.Module):
    def __init__(self, hidden_size: int = 100)-> None:
        super().__init__()
        self.hidden_size: int = hidden_size
        # input_size = 1
        self.lstm: nn.LSTM = nn.LSTM(input_size=1, 
                            hidden_size=self.hidden_size, 
                            batch_first=True)
        self.linear: nn.Linear = nn.Linear(self.hidden_size, 1)
        
    def forward(self, x: Tensor) -> Tensor: 
        # x: (batch, time_step, 1)
        lstm_out, _ = self.lstm(x)
        # 最後の時刻の出力だけ取り出す
        x_last: Tensor = lstm_out[:, -1, :] # shape: (batch, hidden_size)
        out: Tensor = self.linear(x_last)   # shape: (batch, 1)
        return out.squeeze()                # shape: (batch, )
    

# === モデル・損失函数・最適化手法 === #
model: nn.Module = LSTM(hidden_size=100)
criterion: nn.Module = nn.MSELoss()
optimizer: optim.Optimizer = optim.Adam(model.parameters(), lr=0.001)


# === 学習 === #
epochs: int = 10
losses: List[float] = []

for epoch in range(epochs):
    epoch_loss: float = 0.0
    model.train()  # 学習モード

    for X_batch, y_batch in train_loader:   # train_loader: DataLoader[TensorDatase]
        X_batch: Tensor = X_batch.unsqueeze(-1)     # (batch, )
        y_batch: Tensor = y_batch                   # (batch, )

        optimizer.zero_grad()
        output: Tensor = model(X_batch)             # (batch, )
        loss: Tensor = criterion(output, y_batch)
        loss.backward() # 誤差逆伝播
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss: float = epoch_loss / len(train_loader)
    losses.append(avg_loss)
    print(f"Epoch [{epoch+1}/{epochs}, Loss: {avg_loss:.6f}]")


import matplotlib.pyplot as plt

# 学習損失グラフ
plt.figure(figsize=(8, 5))
plt.plot(losses, label="Training Loss", color='blue')
plt.xlabel("Epoch")
plt.ylabel("Loss (MSE)")
plt.title("Training Loss over Epochs")
plt.legend()
plt.grid()
plt.show()


# === 予測 ===
model.eval()
predictions: List[float] = []
actuals: List[float] = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.unsqueeze(-1)  # (batch, time_step, 1)
        output = model(X_batch)
        predictions.extend(output.numpy())
        actuals.extend(y_batch.numpy())

# 逆正規化
predictions_inv = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
actuals_inv = scaler.inverse_transform(np.array(actuals).reshape(-1, 1))

# === グラフ ===
plt.figure(figsize=(10, 6))
plt.plot(actuals_inv, label="Actual", color='black')
plt.plot(predictions_inv, label="Predicted", color='red')
plt.xlabel("Time step")
plt.ylabel("Nikkei 225 Close Price")
plt.title("Actual vs Predicted on Test Set")
plt.legend()
plt.grid()
plt.show()
