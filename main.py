# main.py

from data.data_loader import prepare_datasets
from model.lstm_model import LSTMModel
from controller.trainer import train_model
from view.plotter import plot_loss_curve, plot_predictions

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def main() -> None:
    """
    株価予測モデルの全体実行関数。
    データ取得、モデル構築、学習、予測、可視化を統括する。
    """
    # --- ハイパーパラメータ定義 ---
    start_date: str = "2010-01-01"
    end_date: str = "2024-09-30"
    split_ratio: float = 0.85
    time_step: int = 60
    batch_size: int = 32
    hidden_size: int = 100
    learning_rate: float = 0.001
    epochs: int = 10
    lambda_phy: float = 0.1
    mu: float = 0.001

    # --- データ準備 ---
    train_dataset, test_dataset, scaler = prepare_datasets(
        start=start_date,
        end=end_date,
        split_ratio=split_ratio,
        time_step=time_step
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # --- モデル・損失関数・最適化 ---

    model = LSTMModel(input_size=1, hidden_size=hidden_size, output_size=1)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # --- 学習 ---
    print("train_model start")
    losses = train_model(
        model=model,
        train_loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        epochs=epochs,
        use_pinn=True,
        lambda_phy=lambda_phy,
        mu=mu
    )
    print("train_model start")
    plot_loss_curve(losses)

    # --- 予測 ---
    model.eval()
    predictions = []
    actuals = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.unsqueeze(-1)
            output = model(X_batch)
            predictions.extend(output.numpy())
            actuals.extend(y_batch.numpy())

    predictions_inv = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    actuals_inv = scaler.inverse_transform(np.array(actuals).reshape(-1, 1))

    plot_predictions(actuals=actuals_inv, predictions=predictions_inv)


main()