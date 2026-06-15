import random
from nn0 import Value, linear

class MiniLanguageModel:
    def __init__(self, vocab_size, embedding_dim):
        self.block_size = 4  # 最大上下文長度
        self.n_layer = 1     # 為了符合 gd 函數的結構
        
        # 1. 詞嵌入矩陣 (Embedding Matrix): 每個詞對應一個向量
        # 這裡用隨機微小值初始化，並包裝成 Value
        self.embedding = [
            [Value(random.uniform(-0.1, 0.1)) for _ in range(embedding_dim)]
            for _ in range(vocab_size)
        ]
        
        # 2. 輸出線性層權重 (Output Linear Layer Weights): 矩陣乘法用 W @ x
        # 輸出維度必須是 vocab_size，這樣才能映射回預測的字
        self.W_out = [
            [Value(random.uniform(-0.1, 0.1)) for _ in range(embedding_dim)]
            for _ in range(vocab_size)
        ]
        
    def get_params(self):
        """收集所有需要被 Adam 優化的參數"""
        params = []
        for row in self.embedding:
            params.extend(row)
        for row in self.W_out:
            params.extend(row)
        return params

    def __call__(self, token_id, pos_id, keys, values):
        """
        前向傳播 (Forward Pass)
        雖然 gd 傳入了 keys 和 values (Transformer 用的 KV Cache)，
        我們這個超簡化模型暫時不需要用到它們。
        """
        # 查表得到當前 token 的向量表示
        x = self.embedding[token_id]
        
        # 通過線性層計算每個單字的 logits (未歸一化的機率)
        logits = linear(x, self.W_out)
        return logits
