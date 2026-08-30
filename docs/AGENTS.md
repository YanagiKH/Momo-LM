# 受限代理工作

Momo-LM 的代理是 deterministic local task runner。它將少量明確 goal forms 轉成白名單工具步驟，保存狀態，並在寫入或訓練前等待一次性核准。它不是能任意規劃、上網或控制電腦的通用自治代理。

## Profiles 與 capabilities

| Profile | 預設 capabilities | 可額外要求 |
|---|---|---|
| `training` | `knowledge.read`, `runtime.inspect` | `model.train` |
| `coding` | `knowledge.read`, `runtime.inspect`, `workspace.read` | `workspace.write` |
| `workplace` | `knowledge.read` | `workspace.read`, `workspace.write` |
| `copilot` | `knowledge.read`, `runtime.inspect`, `workspace.read` | `workspace.write`, `model.train` |

要求 profile 不允許的 capability 會在建立工作時失敗。Capability 是應用層允許清單，不是作業系統 sandbox。

Capability 只設定上限，不會自行增加 plan steps。目前只有 `training` 的 `train:` goal 會選擇 `model.train`，只有 `coding` 的 `write:` goal 會選擇 `workspace.write`。`workplace` 和 `copilot` 目前即使宣告可選寫入 capability，planner 仍只建立唯讀／draft steps。

## 工具

| Tool | Capability | 改變狀態 | 限制 |
|---|---|---:|---|
| `inspect_runtime` | `runtime.inspect` | 否 | 回傳模型、後端與知識庫摘要 |
| `search_knowledge` | `knowledge.read` | 否 | 最多 20 筆，每段最多 4,000 chars |
| `list_files` | `workspace.read` | 否 | 只列 agent workspace，最多 500 筆 |
| `read_text_file` | `workspace.read` | 否 | UTF-8 text，最多 256 KiB |
| `draft_text` | `knowledge.read` | 否 | 產生固定格式的本機草稿，不會傳送 |
| `training_guidance` | `runtime.inspect` | 否 | 回傳固定訓練檢查清單 |
| `write_text_file` | `workspace.write` | 是 | 一個 workspace-relative text file，最多 1 MiB |
| `train_text` | `model.train` | 是 | 使用 goal 中明確文字，最多 100,000 chars、1–10 epochs |

以下工具類型明確禁止且沒有後備實作：`browse_web`、`download`、`drive_vehicle`、`execute_program`、`open_url`、`run_command`、`send_email`、`shell`、`use_camera`、`use_microphone`。

代理不能使用 `momo crawl`。Crawler 只可由使用者從 CLI、Web 資料頁或 Python API 明確呼叫。

## Planner 接受的目標

Planner 故意只支援幾種格式。

### Training

一般 goal 會檢查 runtime 並產生訓練清單：

```bash
momo agent run "準備客服資料訓練檢查" --profile training
```

只有 `train:` 前綴會建立 `train_text` 步驟，而且必須明確要求 capability：

```bash
momo agent run "train: User: 你好\nMomo: 你好，請問需要什麼資訊？" \
  --profile training \
  --capability model.train
```

這個工作會停在 `waiting_approval`，不會立即更新權重。

### Coding

讀取一個明確檔案：

```bash
momo agent run "read: src/example.py" --profile coding
```

寫入一個檔案的格式是第一行 `write: relative/path`，後面是完整內容：

```bash
momo agent run $'write: notes/plan.md\n# Plan\nRun the tests.' \
  --profile coding \
  --capability workspace.write
```

其他 coding goal 只列出 workspace 並產生 implementation brief。它不解析程式、執行測試或套用 diff。

### Workplace 與 copilot

`workplace` 搜尋本機 knowledge 後產生草稿；不會寄信、發文或更新外部服務。`copilot` 另外檢查 runtime 與 workspace。兩者都只處理本機資料。

## 核准流程

Mutating tool 在執行前建立 approval record，包含：

- agent ID 與 step index
- tool 名稱
- canonical arguments 與 SHA-256
- 原因、建立時間與過期時間

核准只適用於該 agent 的該 step、tool 與完全相同 arguments，而且只能消耗一次。預設 900 秒過期，可在 config 設為 30–86,400 秒。

```bash
momo agent status AGENT_ID
momo agent approve AGENT_ID APPROVAL_ID
```

如果 arguments、step 或 ID 不符，核准會失敗。沒有 wildcard approval 或 profile-wide approval。

## Budgets

| Budget | 預設 | 最小 | 最大 |
|---|---:|---:|---:|
| `max_steps` | 8 | 1 | 128 |
| `max_tool_calls` | 8 | 0 | 128 |
| `max_input_chars` | 12,000 | 256 | 1,000,000 |

CLI 可覆蓋：

```bash
momo agent run "read: README.md" \
  --profile coding \
  --max-steps 2 \
  --max-tool-calls 2 \
  --max-input-chars 2000
```

工具本身另有檔案、字串與查詢上限。Budgets 不提供 CPU、記憶體或程序級隔離；若需要強隔離，應在低權限 OS 帳號、容器與資源限制下啟動整個 Momo-LM。

## 狀態、事件與取消

狀態：`pending`、`running`、`waiting_approval`、`completed`、`failed`、`cancelled`。

```bash
momo agent list --limit 20
momo agent status AGENT_ID
momo agent events AGENT_ID --after 0 --limit 100
momo agent cancel AGENT_ID
```

每個步驟、核准、取消、錯誤與回復都追加到 SQLite event table。常見 credential key 與字串模式會在保存前 redaction，但這是降低意外洩漏的後備措施，不是 secret scanner。不要把 secret 放進 goal、workspace 或訓練文字。

取消是 cooperative：runner 在步驟間檢查狀態。已進入 NumPy 或檔案系統呼叫的單一步驟不一定能立即中斷。

## 程序重啟

代理資料庫使用 SQLite WAL 與 `synchronous=FULL`。啟動時：

- 中斷的唯讀步驟會重設為 pending，然後安全重跑。
- 中斷中的 mutating 或未知工具不會自動重播；工作標記為 failed。
- 等待核准的工作保留原狀，直到核准過期、被取消或由使用者處理。

這避免程序在不知道先前寫入是否完成時再次執行。它不保證跨機器 failover。

## Workspace 邊界

代理只讀寫 config 中的 `agent_workspace_path`，預設 `~/.momo-lm/agent-workspace/`。路徑必須是相對路徑；每次讀寫前解析 symlink 並確認結果仍在 workspace。寫檔使用同目錄臨時檔、`fsync` 與原子替換。

Mods 不受這個邊界限制。啟用 Mod 後，它仍具有 Momo-LM 程序的全部使用者權限。

## Python API

```python
import momo_lm

with momo_lm.load_model("./momo-data") as model:
    agent = model.create_agent(
        "write: reports/summary.md\n# Draft\nReview before sending.",
        profile="coding",
        capabilities=["workspace.write"],
        budgets={"max_steps": 3, "max_tool_calls": 3},
    )
    if agent["status"] == "waiting_approval":
        approval = agent["pending_approval"]
        agent = model.approve_agent(agent["id"], approval["id"])
    print(agent["status"])
    print(model.agent_events(agent["id"]))
```

`background=True` 會在程序內 daemon thread 執行；它不是獨立 worker service。關閉 `MomoLM` context 時會等待仍受管理的 threads 結束。

## HTTP API

| Method | Path | 行為 |
|---|---|---|
| `GET` | `/api/agents?limit=100` | 列出工作 |
| `GET` | `/api/agents/{id}` | 讀取一筆工作 |
| `GET` | `/api/agents/{id}/events?after=0&limit=100` | 讀取追加事件 |
| `POST` | `/api/agents` | 建立 background 工作 |
| `POST` | `/api/agents/{id}/approve` | 消耗 `approval_id` |
| `POST` | `/api/agents/{id}/cancel` | 要求取消 |

建立 payload：

```json
{
  "goal": "read: README.md",
  "profile": "coding",
  "capabilities": [],
  "budgets": {
    "max_steps": 2,
    "max_tool_calls": 2,
    "max_input_chars": 2000
  }
}
```

非 loopback 部署需要 access token；HTTP 防護詳見 [../SECURITY.md](../SECURITY.md)。
