import datetime
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

#ターゲットには複数銘柄も指定可能です
target = "GOOGL"

#1日毎に5日間データを取得します
data = yf.download(target , period='5d', interval = "1d")
print(data)
