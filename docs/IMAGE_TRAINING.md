# 圖像模型與訓練

Momo-LM 的 TinyCanvas v2 是 3,963 參數的提示詞條件座標網路。它用來展示本機圖像資料驗證、風格標籤 conditioning 的程式路徑、解析 backprop、checkpoint 與 tiled rendering。它不是 diffusion model，也沒有市場級動漫、漫畫、插畫或攝影寫實品質的評測證據。

## 模型規格

| 項目 | 值 |
|---|---:|
| Prompt features | 64 |
| Latent | 24 |
| Hidden | 64 |
| Style embeddings | 4 × 24 |
| Parameters | 3,963 float32 |
| Output size | 每邊 128–2048 px |
| Render tile | 32–512 px |

Styles：`anime`、`manga`、`illustration`、`realistic`。這些是 conditioning labels；`realistic` 不代表照片品質。

`quality` API 參數是 supersampling steps 的預設名稱，不代表經過人類評測的畫質等級：

| API value | Supersampling steps |
|---|---:|
| `draft` | 1 |
| `standard` | 2 |
| `high` | 4 |

也可明確指定 1–8 steps。在同一個已記錄的執行環境內，生成流程設計為 deterministic；跨 CPU、BLAS、NumPy、編譯器或平台可能出現浮點差異，因此不保證逐像素一致。Tile size 只改變批次記憶體，不應改變輸出。

## 生成

CLI：

```bash
momo image "雨夜車站的漫畫分鏡" \
  --style manga \
  --negative-prompt "文字, 浮水印" \
  --quality high \
  --output station.png \
  --width 512 --height 512 --seed 42
```

Python：

```python
from momo_lm.image_model import TinyCanvasModel

model = TinyCanvasModel.load("momo-image-base.npz")
image = model.generate(
    "午後窗邊的角色插畫",
    width=512,
    height=512,
    seed=42,
    style="illustration",
    negative_prompt="文字浮水印",
    quality="standard",
    tile_size=128,
)
image.save("character.png")
```

## Manifest

圖像訓練只接受 format v1 UTF-8 JSON manifest：

```json
{
  "format_version": 1,
  "examples": [
    {
      "image": "images/panel-001.png",
      "prompt": "black and white manga portrait panel",
      "style": "manga",
      "negative_prompt": "watermark low quality",
      "source": "studio-dataset://panel-001",
      "license": "CC-BY-4.0",
      "sha256": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
    }
  ]
}
```

每筆必填 `image`、`prompt`、`style`、`source`、`license`、`sha256`；`negative_prompt` 可省略。`image` 必須是 manifest 目錄內的相對路徑。

Loader 會拒絕：

- 絕對路徑、`..` 或 symlink 逸出
- 不存在、空白或超過 64 MiB 的影像
- SHA-256 不符或在載入期間改變的檔案
- 非 PNG／JPEG／WebP、解碼失敗或超過 20,000,000 pixels
- 未知欄位、缺少來源／授權、未知 style
- 超過 2 MiB 的 manifest、超過 10,000 筆 examples
- 單一文字欄位超過 16 KiB UTF-8

`license` 欄位只記錄操作者的聲明。Loader 不會替你判斷授權是否真實或彼此相容。

## 訓練 API

```python
from pathlib import Path

from momo_lm.image_model import TinyCanvasModel
from momo_lm.image_training import load_manifest, train_manifest

model = TinyCanvasModel.load(Path("momo-image-base.npz"))
manifest = load_manifest(Path("dataset/manifest.json"))
report = train_manifest(
    model,
    manifest,
    epochs=8,
    learning_rate=0.02,
    samples_per_image=512,
    seed=20260830,
    gradient_clip=1.0,
)
model.save(Path("checkpoints/image-domain-v1.npz"))
print(report.to_dict())
```

`train_manifest` 對每張影像抽樣座標，計算 RGB reconstruction loss，並以手寫 NumPy gradient 更新全部七個張量。Training report 包含：

- `initial_loss`、`final_loss`
- `steps`、`examples`、`manifest_examples`
- `epochs`、`samples_per_image`
- `manifest_sha256`
- `per_style_initial_loss`、`per_style_final_loss`

每次訓練前先複製 checkpoint，完成後在未參與訓練的 images 上評估。不要只看 train loss。

## 建立流程測試資料

專案提供 deterministic CC0 procedural reference set，供測試 pipeline，不是產品訓練集：

```python
from pathlib import Path
from momo_lm.image_training import create_reference_manifest

manifest_path = create_reference_manifest(Path("reference-set"), size=64)
print(manifest_path)
```

它會建立四張 64×64 PNG 與 manifest，每種 style 一張。因資料極少且程序化，loss 只能用來確認 optimizer 與 conditioning 路徑工作。

## 隨附 checkpoint

隨附圖像 checkpoint 的可重現設定：

| 項目 | 值 |
|---|---:|
| Unique images | 4 |
| Epochs | 24 |
| Optimizer steps | 96 |
| Examples processed | 96 |
| Samples per update | 512 |
| Learning rate | 0.5 |
| Seed | 20260830 |
| Initial mean loss | 0.1402371544 |
| Final mean loss | 0.0728892800 |

各 style 在相同固定座標樣本的 loss：

| Style | Initial | Final |
|---|---:|---:|
| anime | 0.0874478221 | 0.0338756777 |
| manga | 0.2420788258 | 0.1992309839 |
| illustration | 0.1281192750 | 0.0307133999 |
| realistic | 0.1033026949 | 0.0277370587 |

Checkpoint SHA-256：`92ccb5f37a946bcc478f8cccca3d2b7edb513d51233061c05a40b7d298b16b7c`

Manifest SHA-256：`f486d944d01277acdb30b7de7cc428bf98be890e376de996a163ca2d60e90229`

這些數字是在同一個四張圖資料上訓練與評估，沒有 held-out image。在 style labels 作為輸入時，該次訓練降低了 reference set 的 reconstruction loss；每個 label 只對應一張圖，因此不能隔離或證明 style conditioning 的效果。這些數字也不衡量構圖、prompt alignment、FID 或人類偏好。

## 自訂資料建議

1. 確認每張影像的作者、來源、授權與是否允許衍生模型。
2. 先用感知 hash／embedding 在 split 前去除近似重複。
3. 讓同一作品、角色、連續畫格與裁切版本只出現在一個 split。
4. 各 style 保留相近數量與解析度分布。
5. 保存原始檔 SHA-256；資料改動後建立新 manifest。
6. 用 held-out prompt-image pairs 報告 per-style loss。
7. 加入盲測的人類評估，明確定義 prompt alignment、構圖與 artifact rubric。

TinyCanvas 容量有限。增加資料量不會自動突破 3,963 參數的表達上限；若目標是大型 diffusion 品質，應把經授權的本機 diffusion engine 做成獨立 Mod，並另行處理模型授權、硬體需求與安全審查。
