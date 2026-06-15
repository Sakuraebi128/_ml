# 🧠 nn0.py：從零打造的微型深度學習引擎

本專案包含了一個純 Python 實現的**自動微分引擎 (Autograd)** 與 **Adam 優化器**。它不依賴 PyTorch 或 TensorFlow 等大型框架，僅用基礎數學與 Python 語法，完整還原了現代大型語言模型（LLM）最核心的底層運作機制。

---

## 🚀 核心元件拆解

這個引擎主要由以下四大模組構成：

### 1. 基礎節點：`Value` 類別
`Value` 是整個引擎的靈魂。它不僅儲存了數值（`data`），還會儲存該數值的梯度（`grad`）以及它是怎麼被計算出來的（`_children`）。
* **自動追蹤軌跡**：每次進行加（`+`）、乘（`*`）、次方（`**`）或 `relu()` 等運算時，它都會在幕後悄悄建立一張「計算圖（Computation Graph）」。
* **反向傳播（`backward`）**：當我們對最後的 Loss 呼叫 `.backward()` 時，它會使用**微積分的連鎖律（Chain Rule）**，從後往前把梯度一路傳回最前面的輸入參數。

### 2. 神經網路基本運算
* **`linear(x, w)`**：矩陣乘法 $y = W \cdot x$。這是所有神經網路層（如全連接層、Transformer 的 Linear 層）的核心。
* **`rmsnorm(x)`**：RMS Normalization（均方根歸一化）。現代大語言模型（如 LLaMA）普遍採用它來取代傳統的 LayerNorm，用來穩定深層網路的訊號。
* **`softmax(logits)` / `cross_entropy(...)`**：
  * `softmax` 將網路輸出的分數轉換成機率分佈。
  * `cross_entropy` 則計算模型預測與真實標籤之間的差距（Loss）。程式碼中特別使用了 **Log-Sum-Exp 技巧** 來確保計算時不會因為數值過大或過小而崩潰（數值穩定性）。

### 3. 大腦的升級器：`Adam` 優化器
光有梯度還不夠，我們需要調整參數。`Adam` 是目前深度學習最常用的優化器。
* 它會同時紀錄梯度的「一階動量（方向）」與「二階動量（步長變異數）」。
* 能夠自動為每個參數動態調整學習率（Learning Rate），讓模型收斂得又快又穩。

### 4. 訓練核心：`gd` 函數
這是一個標準的**梯度下降訓練迴圈**（Training Loop）：
1. **Forward（前向傳播）**：把字詞輸入模型，計算預測結果。
2. **Loss（計算損失）**：比對預測結果與下一個正確單字的差距。
3. **Backward（反向傳播）**：計算所有參數的梯度。
4. **Update（參數更新）**：優化器根據梯度調整參數，並清除舊梯度。

---

## 📖 實戰範例：讓模型學會講一句話

為了測試這個引擎，我們設計了一個 **Next-Token Prediction（下一詞預測）** 的極簡任務。我們將建立一個超輕量模型，讓它學會說：`"i love neural networks"`。

### 1. 模型架構 (`model.py`)
我們寫一個最簡單的模型，包含「詞嵌入層（Embedding）」與「輸出線性層（Linear）」：

```python
import random
from nn0 import Value, linear

class MiniLanguageModel:
    def __init__(self, vocab_size, embedding_dim):
        self.block_size = 4
        self.n_layer = 1
        
        # 詞嵌入矩陣：每個單字對應一個隨機初始化的向量
        self.embedding = [[Value(random.uniform(-0.1, 0.1)) for _ in range(embedding_dim)] for _ in range(vocab_size)]
        # 輸出線性層：將向量映射回所有單字的機率空間
        self.W_out = [[Value(random.uniform(-0.1, 0.1)) for _ in range(embedding_dim)] for _ in range(vocab_size)]
        
    def get_params(self):
        # 收集所有需要更新的 Value 節點
        params = []
        for row in self.embedding: params.extend(row)
        for row in self.W_out: params.extend(row)
        return params

    def __call__(self, token_id, pos_id, keys, values):
        x = self.embedding[token_id]
        return linear(x, self.W_out)
