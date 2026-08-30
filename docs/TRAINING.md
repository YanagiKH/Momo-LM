# 文字模型訓練

Momo-LM 的文字模型只有 223,835 個參數。它適合檢查資料、反向傳播、checkpoint 與小型領域調整，不適合靠大量重複 epochs 追求大型聊天模型的能力。

## 先決定要訓練還是檢索

| 需求 | 建議 |
|---|---|
| 需要來源、經常更新的規則或產品資料 | `momo ingest` |
| 固定語氣、句型、分類或釐清方式 | `momo train` |
| 大量文件 | 全部 ingest，只挑人工審查的對話 train |
| 未知品質的網站 | crawl 但不要加 `--train` |
| 密碼、個資、機密或無授權內容 | 不要匯入或訓練 |

檢索資料可以刪除或更新，不會立即改變生成分布。權重訓練會影響所有回答，必須保存起始 checkpoint。

## 建立隔離實驗

不要直接在唯一的正式 home 測試。每個實驗使用不同目錄：

```bash
momo --home ./experiments/support-v1 init
momo --home ./experiments/support-v1 inspect > experiments/support-v1/baseline.json
```

保留下列資訊：

- Git commit 與 Python／NumPy 版本
- 原生後端與 CPU／作業系統
- config 完整副本
- 每個資料檔的來源、授權與 SHA-256
- train／validation／challenge 切分清單
- 起始與結束 checkpoint SHA-256
- 完整命令、seed、optimizer steps 與 metrics

## 資料切分

可從這個比例開始：

| Split | 比例 | 是否更新權重 |
|---|---:|---:|
| train | 70–80% | 是 |
| validation | 10–15% | 否 |
| challenge | 10–15% | 否 |

同一文件的相鄰段落、同一問答的改寫或同一模板產生的樣本必須留在同一 split。否則 validation loss 會因內容洩漏而偏低。

專案隨附的 `evals/base-model-report.json` 用於 checkpoint 回歸測試。它的 validation 來自固定 starter corpus，與訓練 corpus 有重疊，不是獨立 held-out evaluation。不要用該數字比較其他模型。

## 資料格式

模型直接處理 UTF-8 bytes，不需要建立詞表。繁體中文、日文、英文與程式碼可以混合，但每批資料應有單一可說明的目的。

知識文件：

```text
[來源] support-policy-v3
[日期] 2026-08-01
[主題] 退款期限
完成付款後七日內可提出申請……
```

對話：

```text
User: 這張訂單能退款嗎？
Momo: 我需要購買日期、商品類型與使用狀態才能判斷。
User: 購買後三天，數位內容尚未啟用。
Momo: 依 support-policy-v3，這筆訂單仍在七日期限內；請再由負責人確認例外條款。
```

資料中要包含「資訊不足時反問」、「文件沒有答案」與「需要人工確認」的例子。不要只收集肯定回答。

## 清理檢查

訓練前逐項檢查：

1. 移除密碼、token、email、電話、地址與未必要的個資。
2. 刪除重複段落與模板殘留。
3. 標記互相衝突的版本，不要混成一個無來源答案。
4. 確認資料與產生的權重可依授權再散布。
5. 將同來源內容分組後再切 split。
6. 用固定問題人工檢查編碼、換行與答案品質。

## 訓練循環

```bash
cp ~/.momo-lm/weights/momo-text-base.npz checkpoints/support-v0.npz
momo --home ./experiments/support-v1 ingest data/support-reference.txt
momo --home ./experiments/support-v1 train data/support-dialogues.txt \
  --epochs 3 \
  --learning-rate 0.0005
momo --home ./experiments/support-v1 inspect > checkpoints/support-v1-stats.json
```

訓練器使用 deterministic shuffle、AdamW、weight decay、global-norm clipping 與 replay。相同程式版本、輸入順序、seed 與環境應得到相同結果；不同 BLAS、CPU 指令或 NumPy 版本仍可能有小型浮點差異。

先用少量 epochs。Loss 下降只表示模型更能預測該資料的下一個 token，不表示回答更真實或更安全。

`learning_rate` 的明確 CLI／API override 必須在 `1e-7` 到 `0.005` 之間；超出範圍會拒絕，不會靜默夾值。載入舊 config 時，正有限的舊值會為相容性夾到此範圍。v3 預設是 `0.0005`；對話 self-learning 使用設定值的 20%，預設 `0.0001`。調整後仍需用相同 validation 和 challenge set 比較。

## 評估

至少保存三類結果：

### Token metrics

- mean negative log-likelihood
- perplexity `exp(NLL)`
- token top-1 accuracy
- non-finite logits／gradients 數量

### 行為檢查

- 基本問候與短對話
- 領域內有來源問題
- 領域外問題與拒答
- 資訊不足時反問
- 繁中、日文、英文與錯誤 UTF-8 邊界
- 重複、空白、超長輸入

### 回歸檢查

- 原本可回答的固定問題
- 生成重複率與無效控制字元
- checkpoint 可重新載入
- v1／v2 遷移後輸出 shape
- NumPy 與原生後端數值誤差

只有未參與訓練、去除近似重複的資料才能稱為 held-out。若由人評估，請保存 rubric、盲測順序與每位評估者的原始結果。

## Replay 與災難性遺忘

常見症狀：基本問候退步、只回答單一主題、固定句子重複、亂碼增加或 loss 突然成為非有限值。

處理順序：

1. 停止訓練，保留失敗 checkpoint 與 log 供分析。
2. 回復 last-good 或明確保存的前一版。
3. 降低 learning rate、epochs 或單批重複率。
4. 在 replay 加入基礎對話、拒答與釐清問題。
5. 移除互相矛盾、極長或編碼錯誤的樣本。
6. 把易變事實改放知識庫。

## Checkpoint 驗證與回復

文字 format v3 保存每個張量的 shape、dtype、nbytes 與 SHA-256。載入器會先檢查整份 checkpoint，再改變 runtime 狀態。

```bash
sha256sum checkpoints/support-v0.npz
sha256sum ~/.momo-lm/weights/momo-text-base.npz
momo --home ./experiments/support-v1 inspect
```

若新權重失敗，停止服務後用已驗證 checkpoint 覆蓋實驗 home 內的文字權重。不要編輯 NPZ metadata 來繞過 shape 或 hash 檢查。

## 發布自訂權重

公開 checkpoint 前需附上：

- 模型 shape 與 format version
- 基礎 checkpoint 與最終 checkpoint SHA-256
- 資料集名稱、版本、授權與移除個資方式
- train／validation／challenge 的去重方法
- 訓練命令、seed、optimizer 與 steps
- token metrics 與人工評測原始資料
- 已知失敗案例與適用範圍

不能合法再散布的資料可能仍會被 checkpoint 記住；只刪除原始檔並不能解除風險。
