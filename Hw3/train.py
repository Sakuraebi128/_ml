from nn0 import Adam, gd
from model import MiniLanguageModel

def main():
    # --- 1. 資料準備與 Tokenization ---
    text = "i love neural networks"
    words = text.split()
    
    # 建立字典 (Vocabulary)
    vocab = sorted(list(set(words)))
    word_to_id = {w: i for i, w in enumerate(vocab)}
    id_to_word = {i: w for i, w in enumerate(vocab)}
    vocab_size = len(vocab)
    
    print(f"文本內容: '{text}'")
    print(f"詞彙表: {word_to_id}")
    
    # 將文本轉換成 Token ID 序列
    tokens = [word_to_id[w] for w in words] # 例如: [0, 1, 2, 3]
    
    # --- 2. 初始化模型與優化器 ---
    embedding_dim = 4  # 向量維度
    model = MiniLanguageModel(vocab_size, embedding_dim)
    
    # 取得模型所有參數並放入 Adam 優化器
    params = model.get_params()
    optimizer = Adam(params, lr=0.1) # 給予較大的學習率，因為資料量極小
    
    print(f"模型參數量: {len(params)}")
    print("開始訓練...\n")
    
    # --- 3. 訓練迴圈 ---
    num_steps = 50
    for step in range(num_steps):
        # 呼叫你寫好的 gd 函數進行前向、反向傳播與參數更新
        loss_val = gd(model, optimizer, tokens, step, num_steps)
        
        if (step + 1) % 10 == 0 or step == 0:
            print(f"Step {step+1:02d}/{num_steps} | Loss: {loss_val:.4f}")
            
    # --- 4. 驗證模型預測能力 ---
    print("\n--- 訓練完成，測試模型預測能力 ---")
    
    # 測試給予第一個詞 "i"，看能不能預測出後續整句話
    current_word = "i"
    sentence = [current_word]
    
    # 我們預測 3 次，試圖接完這句話
    for _ in range(3):
        current_id = word_to_id[current_word]
        
        # 虛擬的 KV cache 丟入，雖然模型沒用到
        dummy_keys = [[]]
        dummy_values = [[]]
        
        # 預測下一個字的 logits
        logits = model(current_id, 0, dummy_keys, dummy_values)
        
        # 找出機率最大（data 最大）的 token id
        next_id = max(range(len(logits)), key=lambda i: logits[i].data)
        next_word = id_to_word[next_id]
        
        sentence.append(next_word)
        current_word = next_word
        
    print(f"模型生成的句子: {' '.join(sentence)}")

if __name__ == "__main__":
    main()
