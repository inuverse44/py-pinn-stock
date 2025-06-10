# view/plotter.py

import matplotlib.pyplot as plt
import numpy as np
from typing import List

def plot_loss_curve(losses: List[float]) -> None:
    """
    学習エポックごとの損失（Loss）をプロットする。

    Parameters
    ----------
    losses : List[float]
        各エポックでの平均損失値。
    """
    plt.figure(figsize=(8, 5))
    plt.plot(losses, label="Training Loss", color="blue")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.title("Training Loss over Epochs")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()


def plot_predictions(
    actuals: np.ndarray,
    predictions: np.ndarray
) -> None:
    """
    テストデータに対する予測値と実測値を比較してプロットする。

    Parameters
    ----------
    actuals : np.ndarray
        実際の株価データ（正規化を戻した後の値）。
    predictions : np.ndarray
        モデルによる予測株価（正規化を戻した後の値）。
    """
    plt.figure(figsize=(10, 6))
    plt.plot(actuals, label="Actual", color="black")
    plt.plot(predictions, label="Predicted", color="red")
    plt.xlabel("Time Step")
    plt.ylabel("Nikkei 225 Close Price")
    plt.title("Actual vs Predicted Stock Prices")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()
