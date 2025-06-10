# data/data_loader.py

import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
from torch import Tensor
from torch.utils.data import TensorDataset
from typing import Tuple

def load_nikkei_data(start: str, end: str) -> np.ndarray:
    nikkei = yf.download("^N225", start=start, end=end)
    return nikkei["Close"].values.reshape(-1, 1)

def create_dataset(data: np.ndarray, time_step: int) -> TensorDataset:
    X, y = [], []
    for i in range(time_step, len(data)):
        X.append(data[i-time_step:i, 0])
        y.append(data[i, 0])
    X_tensor: Tensor = torch.tensor(np.array(X), dtype=torch.float32)
    y_tensor: Tensor = torch.tensor(np.array(y), dtype=torch.float32)
    return TensorDataset(X_tensor, y_tensor)

def prepare_datasets(
    start: str, end: str,
    split_ratio: float,
    time_step: int
) -> Tuple[TensorDataset, TensorDataset, MinMaxScaler]:

    raw_data: np.ndarray = load_nikkei_data(start, end)

    scaler: MinMaxScaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data: np.ndarray = scaler.fit_transform(raw_data)

    train_size: int = int(len(scaled_data) * split_ratio)
    train_data = scaled_data[:train_size]
    test_data = scaled_data[train_size - time_step:]

    train_dataset: TensorDataset = create_dataset(train_data, time_step)
    test_dataset: TensorDataset = create_dataset(test_data, time_step)

    return train_dataset, test_dataset, scaler


if __name__ == '__main__':
    
    START_DATE : str = "2010-01-01"
    END_DATE: str = "2024-09-30"
    SPLIT_RETION: float = 0.85
    TIME_STEP: int = 60
    nikkei_data: np.ndarray = load_nikkei_data(start="2010-01-01", end="2024-09-30")
    print(nikkei_data)
    dataset: TensorDataset = create_dataset(data=nikkei_data, time_step=TIME_STEP)
    print(dataset)
    train_dataset, test_dataset, scaler = prepare_datasets(start=START_DATE, 
                                                           end=END_DATE, 
                                                           split_ratio= 0.85,
                                                           time_step=TIME_STEP)
    print(train_dataset, test_dataset, scaler)

    