# Momo-LM 架構

Momo-LM 由可獨立訓練的文字模型、條件圖像網路、檢索記憶、原生運算核心、工具層與兩種介面組成。它不使用雲端 AI SDK，也不需要 API Key。

## 文字模型

輸入會被編碼成固定 259 項詞表：`PAD`、`BOS`、`EOS` 與 256 個 UTF-8 byte。最近 24 tokens 分別取得 32 維 embedding，再依位置串接成 768 維向量。

768 維上下文同時進入主投影與 sigmoid gate。96 個隱藏神經元以每 16 個為一組，依序使用 tanh、近似 GELU 與 SiLU；gate 控制每個神經元的資訊量。全部 embedding 的平均值另經殘差矩陣接到隱藏層，減少深路徑遺失短期語意。最後由 259 維 output layer 與 stable softmax 產生下一 token 機率。

模型共有 184,131 個可訓練參數。訓練包含完整的 mixed-activation derivative、gate derivative、殘差 gradient、cross-entropy、global-norm gradient clipping 與 mini-batch SGD。

## 原生運算路徑

`TensorBackend` 將模型矩陣工作路由到 Rust、C++ 或 NumPy。Rust shared library使用經尺寸檢查的 slice 實作；C 提供 cache-blocked matmul、softmax 與 layer norm；C++ 把投影、gate、混合激活和殘差融合成神經元組 executor。CPython bridge 透過 buffer protocol 取得連續 float32 張量並在計算時釋放 GIL。

預設選擇順序是 Rust → C++ → NumPy。所有後端遵守相同 ABI、形狀與容許誤差測試。詳細建置、環境變數與 C ABI 見 [NATIVE_CORE.md](NATIVE_CORE.md)。

## 權重相容性

- `momo-text-base.npz`：文字張量與 JSON metadata。
- `momo-image-base.npz`：TinyCanvas 張量與 shape metadata。
- format v1 會在載入時補上 gate 與 residual 張量；舊權重不需重新下載。
- 下一次保存會寫成 format v2，包含完整 184,131 個參數。
- `.npz` 一律使用 `allow_pickle=False`，checkpoint 不會在載入時執行 Python 物件。

## 回答路徑

1. `ModManager` 執行 `before_chat`，並檢查 `/command`。
2. `KnowledgeStore` 使用關鍵詞與 CJK bigram 在 SQLite 文件中檢索。
3. 命中基礎問答時直接回覆已驗證答案；命中使用者資料時組合來源證據。
4. 沒有命中時由 gated neuron model 產生並過濾無效控制字元。
5. 資訊過於含糊時先提出具體釐清問題。
6. 執行 `after_chat` 並保存對話；若允許自我學習，再以低學習率更新權重。

## Python 嵌入 API

`momo_lm.api.MomoLM` 是其他應用的穩定 façade。它負責權重初始化、runtime 生命週期、訓練、資料匯入、圖像、語音與狀態，不要求呼叫端直接操作 server 或內部模型。

```python
import momo_lm

with momo_lm.load_model("./runtime") as model:
    print(model.chat("Hello", learn=False))
```

專案發行名保留 `Momo-LM`，Python import 使用合法識別字 `momo_lm` 或相容模組 `MomoLM`。

## Web 工作台

`ThreadingHTTPServer` 提供單頁介面與 JSON endpoints。服務預設只監聽 `127.0.0.1`，回應包含 `nosniff`，靜態與生成檔案都以解析後路徑限制在指定 root，避免目錄穿越。

## 圖像與語音

TinyCanvas 把提示詞轉成 64 維雜湊特徵與 latent，並將每個像素座標、週期特徵和 latent 送入小型 MLP 產生 RGB。語音使用作業系統的離線 TTS；Momo fallback tones 僅確保在無 TTS 套件時流程仍能輸出 WAV。
