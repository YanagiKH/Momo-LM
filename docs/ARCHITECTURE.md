# Momo-LM 架構

Momo-LM 由一個可獨立訓練的文字模型、條件圖像網路、檢索記憶、工具層與兩種介面組成。核心只依賴 NumPy 和 Pillow，沒有雲端 AI SDK。

## 文字模型

輸入先被編碼成固定 259 項詞表：`PAD`、`BOS`、`EOS` 與 256 個 UTF-8 byte。最近 24 tokens 分別取得 32 維 embedding，再依位置串接成 768 維向量。它經過 96 維 tanh hidden layer 與 259 維 output layer，softmax 得到下一 token 機率。

訓練使用手動實作的 cross-entropy gradient、反向傳播、global-norm gradient clipping 與 mini-batch SGD。這使核心容易閱讀和移植，但品質及效率不等同於深層 Transformer。

## 回答路徑

1. `ModManager` 執行 `before_chat`，並檢查 `/command`。
2. `KnowledgeStore` 使用關鍵詞與 CJK bigram 在 SQLite 文件中檢索。
3. 命中基礎問答時直接回覆經驗證答案；命中使用者資料時附來源組合證據。
4. 沒有命中時由 `NeuralTextModel` 生成，並過濾無效控制字元。
5. 資訊過於含糊時先提出具體釐清問題。
6. 執行 `after_chat`，保存對話；若允許自我學習，再以低學習率更新權重。

## 權重與資料

- `momo-text-base.npz`：文字模型張量與 JSON metadata。
- `momo-image-base.npz`：TinyCanvas 張量與 shape metadata。
- `momo.db`：文件片段與最近 1,000 輪對話。
- `config.json`：主機、生成與學習設定。

`.npz` 讀取明確使用 `allow_pickle=False`，避免 checkpoint 在載入時執行任意 Python 物件。

## Web 工作台

`ThreadingHTTPServer` 提供單頁介面與 JSON endpoints。服務預設只監聽 `127.0.0.1`，回應包含 `nosniff`，靜態與生成檔案都以解析後路徑限制在指定 root，避免目錄穿越。

## 圖像與語音

TinyCanvas 把提示詞轉成 64 維雜湊特徵與 latent，並將每個像素座標、週期特徵和 latent 送入小型 MLP 產生 RGB。語音則使用作業系統的離線 TTS；Momo fallback tones 僅確保在無 TTS 套件時流程仍能輸出 WAV。
