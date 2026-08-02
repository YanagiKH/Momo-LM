# Momo-LM Mod 開發

Mod 是放在執行目錄 `mods/` 的 Python 檔。預設位置是 `~/.momo-lm/mods/`；每個實驗也可用 `momo --home PATH` 擁有自己的 Mods。

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

Mod 不是 sandbox。它能讀寫使用者可存取的檔案、連網與啟動程式。只載入自己撰寫或完整審查過的檔案，不要從不明來源複製 Mod。

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
