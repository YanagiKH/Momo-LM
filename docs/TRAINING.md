# Momo-LM 訓練手冊

本手冊用於建立可重現的個人模型、垂直領域模型與多領域實驗。Momo-LM 的文字模型很小，因此「乾淨資料、保守學習率、固定驗證集」比堆疊大量低品質文字更重要。

## 1. 建立獨立實驗

不要直接在唯一的正式權重上測試。每個實驗使用不同 `MOMO_HOME`：

```bash
momo --home ./experiments/support-bot init
momo --home ./experiments/support-bot inspect > experiments/support-bot/baseline.json
```

所有權重、資料庫、產物與 Mods 都會保存在該目錄。

## 2. 資料分層

建議把資料分成三組：

| 資料 | 比例 | 用途 |
|---|---:|---|
| train | 70–80% | 更新模型權重 |
| validation | 10–15% | 比較每個版本，不更新權重 |
| challenge | 10–15% | 未知問題、歧義、安全邊界與跨語言測試 |

同一篇文章的相鄰段落不能被分散到 train 與 validation，否則評估會因內容洩漏而過度樂觀。

## 3. 資料格式

模型使用 UTF-8 bytes，不需要建立新詞表。可以混合繁中、日文、英文與程式碼，但每一批仍應有清楚目的。

知識文件：

```text
[來源] internal-handbook-v3
[主題] 退款流程
客戶於購買後七日內……
```

對話示範：

```text
User: 客戶在什麼條件下可以退款？
Momo: 根據退款流程 v3，條件包括……
User: 資訊不足時怎麼做？
Momo: 我需要訂單日期與商品類型才能判斷。
```

應保留「資訊不足時反問」與「資料中沒有答案」的範例，避免模型只學會強行回答。

## 4. 檢索與權重訓練的選擇

| 需求 | 建議方式 |
|---|---|
| 精確文件事實、經常更新的政策 | `momo ingest`，只進知識庫 |
| 語氣、常用句型、固定對話習慣 | `momo train` |
| 大量參考文件 | 先 ingest，再選出高品質問答 train |
| 尚未確認品質的網站 | crawl 但不要 `--train` |

知識庫內容可以保留來源，也不會直接改變基礎語言能力；權重訓練會改變生成分布，必須版本化。

## 5. 建議訓練循環

```bash
cp ~/.momo-lm/weights/momo-text-base.npz checkpoints/model-v0.npz
momo ingest domain-reference.txt
momo train domain-dialogues.txt --epochs 3 --learning-rate 0.02
momo inspect > checkpoints/model-v1-stats.json
```

使用固定驗證問題測試 v1。若回答改善且通用測試沒有退步，再進行下一批；否則回復 v0 並修正資料。

## 6. 超參數

- `epochs`：先用 3。資料量少但重複度高時，增加 epochs 容易過擬合。
- `learning-rate`：建議 `0.01–0.03`。增量對話學習自動使用主學習率的 20%。
- `temperature`：`config.json` 預設 `0.78`；降低會更穩定，提高會更多樣。
- `top_k`：預設 32；越小越保守。
- `max_new_tokens`：預設 180；位元組 tokenizer 下，非 ASCII 字元會使用多個 token。

## 7. 垂直領域專家流程

1. 定義明確任務邊界與拒答範圍。
2. 整理術語表、規範文件、決策流程和代表性問答。
3. 先把文件加入檢索記憶。
4. 把高品質答案與釐清問題整理成訓練對話。
5. 分批訓練，每批建立 checkpoint 與變更紀錄。
6. 使用領域專家審查固定 validation 和 challenge 集。
7. 定期重新驗證已更新的政策與來源。

不要把 Momo-LM 的輸出當成醫療、法律、金融或安全關鍵決策的唯一依據。

## 8. 災難性遺忘與回復

症狀包括：基本問候變差、只會回答單一主題、重複句子或亂碼增加。處理方式：

1. 回復上一個 `.npz` checkpoint。
2. 降低學習率與 epochs。
3. 在新資料中混入基礎對話與「不知道」範例。
4. 移除重複或互相矛盾資料。
5. 將易變事實留在檢索知識庫，不更新權重。

## 9. 自動化與可重現性

保留下列資料：Git commit、Momo config、資料來源與雜湊、訓練命令、開始/結束權重、`momo inspect` 輸出、驗證結果。含敏感資訊的資料與權重不應提交到公開 Git 儲存庫。
