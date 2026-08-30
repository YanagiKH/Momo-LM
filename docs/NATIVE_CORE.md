# C、C++ 與 Rust 原生核心

Momo-LM 原生 ABI v2 提供模型推論需要的 float32 kernels。Python 層使用同一個 `TensorBackend` 介面，並以 NumPy 作為數值參考與無編譯器後備。

原生核心減少 Python 層迴圈與大型中間張量，不能增加 checkpoint 的知識或讓小模型自動具備人類推理能力。

## 元件

| 層 | 位置 | 責任 |
|---|---|---|
| C ABI | `native/include/momo_core.h` | ABI v2、狀態碼、buffer／shape contract |
| C kernels | `native/src/tensor.c` | matmul、normalization、RoPE、attention、Q8、sampler |
| C++ runtime | `native/src/runtime.cpp` | fused gated mixed-activation neuron groups |
| CPython bridge | `native/python/module.cpp` | buffer protocol、shape 驗證、GIL 釋放 |
| Rust | `native/rust/src/lib.rs` | 同規格 kernels、checked slice／allocation、C ABI |
| Router | `momo_lm/backend.py` | 探測、ABI 檢查、連續化、後備與狀態 |

`MOMO_CORE_ABI_VERSION` 與 `momo_rust_abi_version()` 都必須回傳 `2`。舊或未知 ABI 不會載入。

## Kernels

| Kernel | 輸入摘要 | 數值策略 |
|---|---|---|
| `matmul` | `[rows, inner] × [inner, columns]` | cache-blocked accumulation |
| `softmax` | row-major 2-D logits | 每列先減最大值 |
| `layer_norm` | rows × columns | row mean／variance + epsilon |
| `rms_norm` | rows × columns + weight | row RMS + epsilon |
| `rope` | Q、K、uint64 positions | float64 phase，輸出 float32 |
| `causal_gqa` | Q／K／V full sequence | online max／sum，不配置 score matrix |
| `decode_attention` | one query + KV cache | causal cache scan |
| `quantize_q8` | float32 rows | symmetric int8 + per-row scale |
| `dequantize_q8` | int8 rows + scales | float32 reconstruction |
| `sample` | logits | temperature、top-k、top-p、seed + counter |
| `neuron_group` | projection、gate、residual | fused tanh／GELU／SiLU groups |

## 後端選擇

`MOMO_BACKEND` 接受：

| 值 | 行為 |
|---|---|
| `auto` | Rust → C++，都不可用時退回 NumPy |
| `hybrid`／`native` | 建立 Rust → C++ → NumPy 的 kernel router，但至少一個 native engine 必須可用 |
| `rust` | 只接受 ABI v2 Rust library |
| `cpp` | 只接受 CPython C++ extension |
| `numpy`／`python` | 只使用 NumPy reference |

預設值是 `auto`：

1. 隨 wheel／executable 安裝的 Rust shared library。
2. CPython C/C++ extension。
3. NumPy reference backend。

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

`MOMO_REQUIRE_NATIVE=1` 要求 C++ 與 Rust 建置成功、至少一個 ABI v2 engine 能載入，並拒絕明確選擇 `numpy`／`python`。Hybrid router 仍保留 NumPy reference 供沒有 native 實作的個別 kernel 使用。CI 與 Release 用此設定防止缺少原生產物的套件被誤當成原生建置成功。一般來源安裝沒有編譯器時，extension 和 Rust library 都是 optional，安裝完成後使用 NumPy。

`MOMO_BUILD_RUST=0` 可在 optional 建置中略過 Rust。它不能和 `MOMO_REQUIRE_NATIVE=1` 組合；required build 會明確失敗。

`MOMO_RUST_LIBRARY` 可指向操作者自行建置的 shared library。它會把任意 native code 載入目前程序，只能指向已驗證且 ABI 相容的檔案。

## 建置

Python 安裝會嘗試建置 CPython extension 和 Rust shared library：

```bash
python -m pip install -e ".[dev]"
python scripts/build_native.py --release
momo backend
```

強制原生建置：

```bash
MOMO_REQUIRE_NATIVE=1 python -m pip install -e ".[dev]"
```

只測 C/C++：

```bash
cmake -S native -B build/native \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_COMPILE_WARNING_AS_ERROR=ON
cmake --build build/native --config Release
ctest --test-dir build/native -C Release --output-on-failure
```

只測 Rust：

```bash
cargo fmt --manifest-path native/rust/Cargo.toml --all -- --check
cargo test --manifest-path native/rust/Cargo.toml --release --locked
cargo clippy --manifest-path native/rust/Cargo.toml --all-targets --locked -- -D warnings
```

Rust crate 不使用外部 dependencies。`Cargo.lock` 仍提交到儲存庫，讓 toolchain 輸入可追蹤。

## Sanitizers

Linux 上可重現 CI 的 AddressSanitizer／UndefinedBehaviorSanitizer：

```bash
cmake -S native -B build/sanitized \
  -DCMAKE_BUILD_TYPE=Debug \
  -DMOMO_ENABLE_SANITIZERS=ON
cmake --build build/sanitized --parallel 2
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
ctest --test-dir build/sanitized --output-on-failure
```

Sanitizer 測試涵蓋 native smoke inputs，不是對所有可能 shape 的證明。

## ABI 資料契約

- 所有張量是 row-major、C-contiguous。
- 浮點張量是 IEEE-754 float32；RoPE positions 是 `uint64_t`；Q8 是 `int8_t`。
- 呼叫端配置並持有所有輸入與輸出 buffers。
- 輸入／輸出不得使用未記錄的 alias；呼叫端需提供對應 shape 的完整長度。
- 尺寸乘法在轉成 bytes 或建立 slice 前檢查溢位。
- 會拒絕 null pointer、零維度、不相容 heads、非正 epsilon／scale、非有限輸入與無法表示的輸出。
- GQA 的 query heads 必須能映射到 key/value heads，rotary dimensions 必須為偶數且不大於 head size。
- Q8 每列使用獨立 scale；全零與極小值列不會產生零或非有限 scale。

直接呼叫 C ABI 時，編譯器無法知道 pointer 背後長度。Python bridge 的 shape 檢查不能保護外部 C 呼叫端。

## 狀態碼

| 值 | 名稱 | 意義 |
|---:|---|---|
| `0` | `MOMO_STATUS_OK` | 成功 |
| `-1` | `MOMO_STATUS_INVALID_ARGUMENT` | pointer、shape 或參數無效 |
| `-2` | `MOMO_STATUS_NUMERIC_ERROR` | 非有限數值或數值狀態無效 |
| `-3` | `MOMO_STATUS_OVERFLOW` | 尺寸或位移計算溢位 |
| `-4` | `MOMO_STATUS_OUT_OF_MEMORY` | 內部必要配置失敗 |

## C ABI 範例

```c
#include "momo_core.h"

float left[4] = {1, 2, 3, 4};
float right[4] = {5, 6, 7, 8};
float output[4] = {0};

int status = momo_matmul_f32(left, right, output, 2, 2, 2);
if (status != MOMO_STATUS_OK) {
    return status;
}
```

Sampling：

```c
float logits[4] = {0.1f, 0.8f, 0.3f, -0.2f};
size_t token = 0;
int status = momo_sample_f32(
    logits, 4, 0.8f, 4, 0.9f,
    42, 0,
    &token
);
```

相同 logits、參數、seed 與 counter 會選出相同 token。這是可重現 sampler，不是密碼學亂數。

## Python bridge

```python
import numpy as np
from momo_lm.backend import get_backend

backend = get_backend()
left = np.ones((2, 3), dtype=np.float32)
right = np.ones((3, 4), dtype=np.float32)
output = backend.matmul(left, right)
print(backend.describe())
```

Bridge 只接受 C-contiguous float32／int8 buffers，並在耗時計算時釋放 GIL。若原生函式回傳錯誤碼，Python 端會轉成明確 exception，不會靜默使用部分輸出。

## 數值驗證

測試使用相同 inputs 比較 NumPy、C++ 與 Rust：

- `matmul`、normalization 與 fused neurons 使用明確 `rtol`／`atol`。
- softmax 每列總和接近 1，且沒有非有限值。
- causal attention 與 decode cache 輸出一致。
- RoPE 測試高 position，避免先轉成 float32 造成 phase 精度遺失。
- Q8 驗證範圍、scale、round-trip error 與 subnormal input。
- sampler 驗證 seed／counter 重現性與 top-k／top-p 邊界。
- 無效與極大 shape 必須回傳錯誤，不能解參考 pointer。

浮點一致性允許小型平台誤差；不應把 bit-for-bit native parity 當成跨 CPU 保證。

## 乾淨 sdist 後備測試

CI 先建立 source distribution，再用不存在的 `CC`／`CXX` 並關閉 Rust 建置。安裝後確認：

- `momo_lm._native` 不存在。
- `_native_libs` 不存在或為空。
- `TensorBackend("numpy")` 可執行。

這個測試從乾淨 sdist 安裝，避免工作目錄內殘留 `.so`、`.dll` 或 `.dylib` 讓後備測試假成功。
