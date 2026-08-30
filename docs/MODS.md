# Momo-LM Mod 開發

Mod 是放在執行目錄 `mods/` 的 Python 檔。預設位置是 `~/.momo-lm/mods/`；每個實驗也可用 `momo --home PATH` 擁有自己的 Mods。Mod 是可信任程式碼擴充，不是受限代理工具，也不是 sandbox。

## 最小模組

```python
from momo_lm.mods import ModSpec


def hello(argument: str) -> str:
    return f"Hello, {argument or 'friend'}"


def register() -> ModSpec:
    return ModSpec(
        name="Hello Mod",
        version="1.0.0",
        description="Adds /hello.",
        commands={"/hello": hello},
    )
```

儲存為 `~/.momo-lm/mods/hello.py` 後，在 UI 的 Mods 頁按「重新載入」，或重新啟動 Momo-LM。

## 對話鉤子

```python
from momo_lm.mods import ModSpec


def before_chat(text: str) -> str:
    return text.strip()


def after_chat(text: str, response: str) -> str:
    return response + "\n\n— custom-domain"


def register() -> ModSpec:
    return ModSpec(name="Domain Style", before_chat=before_chat, after_chat=after_chat)
```

鉤子依檔名排序執行。某個 Mod 載入失敗時會出現在 Mods 頁，不會阻止其他 Mod 或主程式啟動。

## 規則

- 命令名稱必須以 `/` 開始，並只使用英文字母、數字、`_`、`-`。
- handler 接收命令後方的文字並回傳可轉換成字串的結果。
- `before_chat` 接收文字並回傳文字。
- `after_chat` 接收原始文字與目前回答並回傳回答。
- Mod 可使用第三方套件，但必須由使用者自行安裝並處理錯誤。

## 安全

Mod 能讀寫使用者可存取的檔案、連網、啟動程式、讀取環境變數，並呼叫任何已安裝的 Python 套件。代理的 capability、approval、workspace 與禁止工具規則不會限制 Mod。

只載入自己撰寫或逐行審查過的檔案。不要從 issue、聊天訊息或未知網站直接複製 Mod。需要第三方套件時，記錄版本、hash、授權與其原生程式碼；不要在 Mod 中保存 token。

載入錯誤會被隔離並顯示在 Mods 頁，但執行中 handler 的副作用無法由 Momo-LM 回復。涉及寫檔或網路的 handler 應自行加入明確確認、timeout、大小限制與 audit log。

## 測試

```python
import tempfile
from pathlib import Path
from momo_lm.mods import ModManager

with tempfile.TemporaryDirectory() as directory:
    # 將待測 mod 複製到 directory
    manager = ModManager(Path(directory))
    manager.load()
    assert not manager.errors
    assert manager.command("/hello Momo") == "Hello, Momo"
```

測試應另外涵蓋空輸入、超長輸入、handler exception、重複命令、非 UTF-8 檔案與重新載入。CI 不會自動信任或簽署第三方 Mod。
