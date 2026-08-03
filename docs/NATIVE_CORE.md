# Momo-LM 原生核心

Momo-LM 0.2 的矩陣、推理與神經元組核心同時提供 C、C++ 與 Rust 實作。Python 層維持單一 `TensorBackend` 介面，模型與上層功能不需要知道實際使用哪個語言後端。

## 元件

| 層 | 位置 | 責任 |
|---|---|---|
| C ABI | `native/include/momo_core.h` | 固定 ABI、float32 張量形狀與錯誤碼 |
| C kernels | `native/src/tensor.c` | cache-blocked matmul、stable softmax、layer normalization |
| C++ runtime | `native/src/runtime.cpp` | gated neuron groups、tanh/GELU/SiLU 混合激活、殘差融合 |
| CPython bridge | `native/python/module.cpp` | buffer protocol、GIL 釋放、Python 原生擴充 |
| Rust kernels | `native/rust/src/lib.rs` | 溢位檢查、slice 邊界、C ABI、同規格推理核心 |
| Python router | `momo_lm/backend.py` | 自動偵測、輸入連續化、後備與狀態資訊 |

## 後端選擇

預設 `MOMO_BACKEND=auto`，依序嘗試：

1. 隨套件建置的 Rust shared library。
2. CPython C/C++ extension。
3. NumPy reference backend。

可固定後端以重現效能或除錯：

```bash
MOMO_BACKEND=rust momo backend
MOMO_BACKEND=cpp momo backend
MOMO_BACKEND=numpy momo backend
momo benchmark --size 512 --rounds 10
```

Windows PowerShell：

```powershell
$env:MOMO_BACKEND = "rust"
momo backend
```

`MOMO_REQUIRE_NATIVE=1` 會禁止退回 NumPy。CI、安裝程式與 Release 都啟用這個設定，因此發布產物一定實際包含可載入的原生核心。

## 建置

一般 Python 開發安裝會自動嘗試建置 C/C++ 與 Rust：

```bash
python -m pip install -e ".[dev]"
python scripts/build_native.py --release
```

獨立驗證 C/C++：

```bash
cmake -S native -B build/native -DCMAKE_BUILD_TYPE=Release
cmake --build build/native --config Release
ctest --test-dir build/native -C Release --output-on-failure
```

獨立驗證 Rust：

```bash
cargo test --manifest-path native/rust/Cargo.toml --release --locked
cargo clippy --manifest-path native/rust/Cargo.toml --all-targets --locked -- -D warnings
```

如果使用者從原始碼安裝但沒有 C/C++ 或 Rust toolchain，安裝仍會完成並使用 NumPy。發行用建置應設定 `MOMO_REQUIRE_NATIVE=1`，避免產生缺少加速核心的成品。

## 數值與記憶體契約

- 所有公開 kernel 使用 row-major、C-contiguous、IEEE 754 float32。
- 所有尺寸在進入 raw pointer/slice 前驗證，Rust 端額外檢查乘法溢位。
- CPython bridge 使用 buffer protocol，不依賴 NumPy C headers；運算期間釋放 GIL。
- Softmax 先減去每列最大值，避免大型 logits 指數溢位。
- LayerNorm 使用每列統計與可設定 epsilon。
- 原生與 NumPy reference 的矩陣、softmax、layer norm、融合神經元組都必須通過 tolerance 測試。
- `MOMO_CORE_ABI_VERSION` 與 `momo_rust_abi_version()` 目前都是 `1`；不相容版本會拒絕載入。

## 直接使用 C ABI

```c
#include "momo_core.h"

float left[4] = {1, 2, 3, 4};
float right[4] = {5, 6, 7, 8};
float output[4];
int status = momo_matmul_f32(left, right, output, 2, 2, 2);
```

`0` 代表成功，負值代表無效指標、尺寸或數值狀態。呼叫端持有所有輸入輸出記憶體，C ABI 不跨邊界配置或釋放使用者緩衝區。
