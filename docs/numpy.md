# 🚀 NumPy → PyTorch 変換に関する解説まとめ

## ① nparr の形状と構造
Pythonコード例
```
nparr = np.block([[nparr, nparr], [nparr, nparr]]).reshape(1, 1, 4, 4)
```
データ例
```
nparr = [[[[1. 2. 1. 2.]
           [3. 4. 3. 4.]
           [1. 2. 1. 2.]
           [3. 4. 3. 4.]]]]
```

(1, 1, 4, 4)
各次元の意味

|次元	|サイズ	|役割例|
|---|---|---|
|0	|1	|バッチサイズ (Batch)|
|1	|1	|チャネル数 (Channel)|
|2	|4	|高さ (Height)|
|3	|4	|幅 (Width)|

## ② torch.from_numpy(nparr).clone() の意味
コード例
```
input = torch.from_numpy(nparr).clone()
```

### 分解して解説

#### 1️⃣ torch.from_numpy(nparr)

- NumPy 配列 → PyTorch Tensor へ変換
- ただし メモリは共有 → NumPy 側と Tensor 側が 同じメモリを参照している
- → 変更が相互に影響する

#### 2️⃣ .clone()
- 完全なコピー を新しく作る
- メモリを独立させる 
- → NumPy 側の変更が PyTorch Tensor に影響しない
- 安全に PyTorch モデルに入力できる形にする

## 処理の流れまとめ

- NumPy配列(nparr) 
- → torch.from_numpy() 
- → メモリ共有Tensor  
- → .clone() 
= → 独立したPyTorch Tensor(input)


## ③ よくある用途・実用例
- 画像処理 (NumPy で画像を読み込み → CNN モデルに入力)
- データ前処理 (NumPy → PyTorch データパイプライン) 
-データ拡張やテンソル操作 (reshape, tile, block を活用)

## ④ 図解イメージ（テキストベース）

【NumPy 配列】 nparr (shape = [1,1,4,4])
         ↓ torch.from_numpy()
【PyTorch Tensor (共有メモリ)】
         ↓ .clone()
【PyTorch Tensor (独立メモリ)】 input (shape = [1,1,4,4])

## ⑤ 補足
- from_numpy は便利だがメモリ共有に注意！予期せぬバグの原因になる
- .clone() をつければ 安全に独立した Tensor を作れる
- PyTorch の tensor() とは動作が異なる（tensor() は常にコピーを作る）


