# Periodic Hill → Eccentric Annulus GILBM 轉換計畫

**版本:** v1.0
**狀態:** Phase 0 ready, Phase 1-3 需要 Edit1_GILBM 原始碼確認後細化

---

## 0. 架構總覽

### 0.1 兩套系統的座標對應

```
                Periodic Hill                 Eccentric Annulus
  ─────────────────────────────────    ─────────────────────────────────
  程式索引   物理意義      BC             程式索引   物理意義      BC
  ─────────────────────────────────    ─────────────────────────────────
   i (NI)    x spanwise   均勻+週期       i (NI)    x streamwise  均勻+週期+驅動
   j (NJ)    y streamwise 週期+buffer     j (NJ)    y 截面座標1   截面網格 (from .DAT)
   k (NK)    z wall-norm  非均勻+wall     k (NK)    z 截面座標2   截面網格 (from .DAT)
  ─────────────────────────────────    ─────────────────────────────────
  截面平面:  y-z (j-k)                   截面平面:  y-z (j-k)
  讀取方式:  Fröhlich .DAT               讀取方式:  bipolar .DAT
  流動驅動:  y (j) 方向壓力梯度           流動驅動:  x (i) 方向壓力梯度
```

### 0.2 核心洞見：為什麼可以復用

你的 Periodic Hill 程式碼設計為 **y-z 平面讀取任意網格**。Eccentric Annulus 的截面同樣在 y-z 平面，所以：

1. **截面網格讀取機制** → 直接複用（換 .DAT 檔即可）
2. **j-k 方向的 metric terms 計算** → 直接複用（度量張量通用）
3. **i 方向均勻格點 + 週期性** → 直接複用（從 spanwise 變 streamwise）
4. **GILBM 插值機制** → 直接複用（通用曲線座標）

**唯一改動集中在：** 邊界條件 + 驅動方向 + 截面拓撲（封閉環 vs 開放通道）

### 0.3 截面網格規格

來自 bipolar grid 系統的 .DAT 檔：

| 項目 | 值 | 說明 |
|------|-----|------|
| I (環向) | 200 | i=0 與 i=199 幾何重合，360° 封閉 |
| J (徑向) | 80 | j=0 外壁 R₂=3.0, j=79 內壁 R₁=1.0 |
| 外壁 | R₂ = 3.0, 圓心 (0,0) | no-slip wall |
| 內壁 | R₁ = 1.0, 圓心 (0,-1) | no-slip wall |
| 偏心率 | ε = 0.5 | 偏移距離 e = 1.0 |
| det(J) 範圍 | 0.556 ~ 45.0 | 81倍，narrow gap 最密 |
| h 範圍 | 0.75 ~ 6.7 | 9倍變化 |

---

## Phase 0 — 網格讀取與幾何驗證

**目標：** 確認程式能讀入偏心環截面 .DAT，正確計算 metric terms，輸出可視化驗證。
**風險：** 零（不碰 solver）

### 0.1 攜帶檔案

從 `eccentric_annulus_bipolar_grid/` 攜帶這 7 個核心檔案到 Edit1_GILBM 專案：

```
eccentric_annulus_bipolar_grid/
  __init__.py                  # 套件入口
  bipolar_constants.py         # α, β, c 幾何常數
  bipolar_transform.py         # ξ,η ↔ x,y 正反向映射
  grid_metrics.py              # h, J, g_ij, Christoffel 符號
  grid_stretching.py           # Vinokur tanh + GILBM 穩定性
  grid_quality.py              # aspect ratio, 正交性
  grid_io.py                   # Tecplot .DAT 讀寫
```

### 0.2 輸出 .DAT 給 solver 讀取

網格生成系統產出的 .DAT 格式：

```
TITLE     = "Eccentric annulus grid"
VARIABLES = "x corner"
"y corner"
ZONE T="EccentricAnnulus"
 I=200, J=80, K=1,F=POINT
DT=(SINGLE SINGLE)
 x(j=0,i=0) y(j=0,i=0)
 x(j=0,i=1) y(j=0,i=1)
 ...
```

**格式與 Fröhlich .DAT 完全一致** — I/J 維度的 POINT 格式，solver 的 `parse_tecplot_dat()` 可直接讀取。

### 0.3 Metric Terms 驗證清單

在 CUDA solver 讀入網格後，需驗證以下量（與 Python 端交叉比對）：

| 驗證項目 | 計算方式 | 預期結果 |
|---------|----------|---------|
| h (scale factor) | `c / (cos(ξ) + cosh(η))` | 全域 > 0, 範圍 [0.75, 6.7] |
| detJ = h² | `J11*J22 - J12*J21` | 相對誤差 < 1e-10 (conformal) |
| g₁₁ = g₂₂ = h² | metric tensor 對角元 | 相對誤差 < 1e-10 |
| g₁₂ = 0 | metric tensor off-diagonal | 正交性偏差 < 1e-10 |
| 內壁半徑 | sqrt(x² + (y+1)²) at j=79 | = 1.0 ± 1e-9 |
| 外壁半徑 | sqrt(x² + y²) at j=0 | = 3.0 ± 1e-9 |

### 0.4 驗證腳本（Python 端）

```python
from eccentric_annulus_bipolar_grid import BipolarGridGenerator

gen = BipolarGridGenerator(r1=1.0, r2=3.0, eccentricity=1.0)
gen.set_resolution(N_xi=200, N_eta=80)
gen.set_stretching(stretch_eta=1.0)
result = gen.generate()

# 產出 .DAT 供 solver 讀取
gen.export("eccentric_annulus_200x80.dat", format="tecplot")

# 同時匯出 metric terms 用於交叉驗證
gen.export("eccentric_annulus_200x80_metrics.csv", format="csv")
```

### Phase 0 通過標準

- [ ] Python 端 43/43 tests pass
- [ ] .DAT 檔成功產出，格式與 Fröhlich .DAT 一致
- [ ] CUDA solver 能 parse .DAT 不報錯
- [ ] CUDA 計算的 h, detJ 與 Python 端差異 < 1e-6 (single precision)
- [ ] 截面網格可視化正確（環形，內外壁位置對）

---

## Phase 1 — 座標方向與週期性對齊

**目標：** 把 solver 的索引/方向/週期性對齊到 Eccentric Annulus 的物理意義。
**改動範圍：** 索引對應 + 驅動方向，不改物理 BC。

### 1.1 方向映射

```
Periodic Hill 中：
  i → x (spanwise):    NI 格點, 均勻 Δx, 週期性
  j → y (streamwise):  NJ 格點, 均勻 Δy, 週期性 + buffer (有效到 NY-4)
  k → z (wall-normal): NK 格點, 非均勻, k=0 bottom wall

Eccentric Annulus 中：
  i → x (streamwise):  NI 格點, 均勻 Δx, 週期性 + 壓力驅動
  j → (截面 I 方向):   NJ=200, 環向, 封閉週期性 (j=0 ≡ j=NJ)
  k → (截面 J 方向):   NK=80, 徑向, k=0 外壁, k=NK-1 內壁, 雙 no-slip
```

### 1.2 關鍵改動

**1.2.1 i 方向：spanwise → streamwise**

| 項目 | 原 (Periodic Hill) | 新 (Eccentric Annulus) | 改動量 |
|------|-------------------|----------------------|--------|
| 格點分布 | 均勻 | 均勻 | 無 |
| 週期性 BC | 是 | 是 | 無 |
| 壓力驅動 | 無 (y 方向驅動) | 有 (i 方向恆定體積力) | **改** |

**1.2.2 j 方向：streamwise → 環向**

| 項目 | 原 (Periodic Hill) | 新 (Eccentric Annulus) | 改動量 |
|------|-------------------|----------------------|--------|
| 格點分布 | 均勻 | 非均勻（來自 .DAT） | **改讀取** |
| 週期性 BC | 週期 + buffer (到 NY-4) | 幾何封閉週期 (j=0≡j=NJ) | **改 BC** |
| buffer layer | 有 (NY-4 到 NY-1) | 無（幾何自然封閉） | **移除** |

**1.2.3 k 方向：wall-normal → 徑向**

| 項目 | 原 (Periodic Hill) | 新 (Eccentric Annulus) | 改動量 |
|------|-------------------|----------------------|--------|
| 格點分布 | 非均勻 Fröhlich mesh | 非均勻 bipolar mesh | **換 .DAT** |
| k=0 | bottom wall (no-slip) | outer wall (no-slip) | 語意變 |
| k=NK-1 | top boundary (TBD) | inner wall (no-slip) | **Phase 2 改** |

### 1.3 驅動力方向轉移

```
原本 (Periodic Hill):
  F_drive = (dp/dy) 沿 j 方向
  透過 body force 或壓力差驅動 streamwise flow

新 (Eccentric Annulus):
  F_drive = (dp/dx) 沿 i 方向
  i 方向是均勻格點，驅動力直接加在速度 u 上
```

這一步改動很小：找到 force controller kernel，把驅動力的分量從 `j` 方向移到 `i` 方向。

### 1.4 j 方向週期性：從 buffer 到幾何封閉

你的核心問題在這裡。**答案是可以沿用**，而且更簡單：

Periodic Hill 的 j 方向週期性有 buffer layer (NY-6 到 NY-1)，是因為 streamwise 方向的壓力差需要特殊處理。Eccentric Annulus 的環向（j 方向）是 **幾何封閉**（360° 回到原點），所以：

```
原本的週期性:  f(j=0) = f(j=NY-4)   （有 buffer offset）
新的週期性:    f(j=0) = f(j=NJ)     （幾何重合，無 offset）
```

實作方式：

```c
// 環向週期 BC — 比原本更簡單
// j=0 和 j=NJ-1 是幾何同一點（.DAT 中 i=0 ≡ i=199）
// 在 GILBM streaming 中，超出 j 範圍的索引直接 modulo NJ:
int j_periodic(int j, int NJ) {
    return ((j % NJ) + NJ) % NJ;  // 標準 modulo，處理負值
}
```

**注意：** 因為 .DAT 中 i=0 和 i=199 是同一點，如果直接用 200 個點會有重複。solver 需要決定：
- 選項 A：讀取所有 200 點，j=NJ-1 = j=0（ghost node），NJ_effective = 199
- 選項 B：只讀 199 點（去掉最後一個），NJ = 199，純 modulo 週期

**建議選項 A**（保留 ghost），與 Periodic Hill 的 buffer layer 概念一致，改動最小。

### Phase 1 通過標準

- [ ] 驅動力方向正確（i 方向）
- [ ] j 方向週期性正確（封閉環）
- [ ] 截面網格讀入正確（j-k 對應 .DAT 的 I-J）
- [ ] 不施加邊界條件、不啟動流場的情況下，metric terms 在 GPU 上正確

---

## Phase 2 — 邊界條件改寫

**目標：** 把 k=NK-1 從原本的 top BC 改成 no-slip wall，實現雙壁面。
**這是唯一改變物理的步驟。**

### 2.1 Periodic Hill 的 k 方向 BC

```
k = 0:      bottom wall — no-slip (bounce-back 或 interpolated)
k = NK-1:   top boundary — free-slip / symmetry / periodic (取決於你的版本)
```

### 2.2 Eccentric Annulus 的 k 方向 BC

```
k = 0:      outer wall (R₂=3.0) — no-slip wall ← 與原本 k=0 相同，不改
k = NK-1:   inner wall (R₁=1.0) — no-slip wall ← 從 top BC 改為 no-slip
```

### 2.3 改動策略

```
步驟 1: 找到 k=NK-1 邊界條件的 kernel / 函數
步驟 2: 將其改為與 k=0 相同的 no-slip 處理
        （可能是 bounce-back、Bouzidi interpolation、或 IBM）
步驟 3: 確認 k=NK-1 附近的 ghost nodes / buffer 設定正確
```

**GILBM-specific 注意：** 在曲線座標中，no-slip BC 通常透過 interpolated bounce-back 實現，因為 wall 不一定對齊計算格點。對於偏心環，k=0 和 k=NK-1 都對應精確的計算邊界（η=α 和 η=β），所以可以用精確的半步 bounce-back。

### 2.4 壁面法向量

```
k=0 (outer wall):    法向量指向管道內部 (inward)
k=NK-1 (inner wall): 法向量指向管道外部 (outward, 即遠離內管軸心)
```

在 bipolar 座標中：
- `η = α`（outer wall）→ 法向量 = +η 方向
- `η = β`（inner wall）→ 法向量 = -η 方向

### Phase 2 通過標準

- [ ] k=0 和 k=NK-1 都是 no-slip
- [ ] 靜止流場測試：初始化零速度，施加壓力驅動，壁面速度保持為零
- [ ] 壁面剪應力方向正確（沿 streamwise）

---

## Phase 3 — 層流驗證

**目標：** 用層流解析解做定量比較，確認整個 pipeline 數值正確。

### 3.1 解析解

偏心環管道的 fully-developed 層流速度分佈有解析解（Snyder & Goldstein 1965, Piercy et al. 1933）。在 bipolar 座標中：

```
u(ξ, η) = (1/4μ)(dp/dx) * Σ_{n=1}^{∞} A_n * sinh(n(η-α)) * cos(nξ) + base_flow
```

其中 A_n 由邊界條件（內外壁 u=0）決定。此級數收斂快，前 20 項足夠。

### 3.2 驗證條件

| 參數 | 值 |
|------|-----|
| Re_D | 10 ~ 100（低 Re，確保層流） |
| 截面 | R₁=1.0, R₂=3.0, ε=0.5 |
| 驅動力 | 恆定 dp/dx |
| GAMMA | 1.0（穩定性安全範圍） |
| 網格 | 200×80×16（截面×streamwise） |

### 3.3 比較指標

| 指標 | 容許誤差 |
|------|---------|
| 截面速度分佈 u(y,z) vs 解析解 | L₂ < 1e-3 |
| 最大速度位置 | 偏向 wide gap 側（解析解可算出精確位置） |
| 壁面摩擦係數 f | 與 Snyder-Goldstein 理論值差 < 5% |
| 質量守恆 | Σ(ρu) 逐時間步偏差 < 1e-10 |

### 3.4 已知 Eccentric Annulus 層流特徵

作為 sanity check：

1. 最大速度位於 wide gap 側（y > 0 區域），偏離中心線
2. 隨 ε 增大，narrow gap 流速急劇降低
3. ε → 0 時退化為同心環流（有精確 Poiseuille 解）
4. 截面流量 Q ∝ (1 + 3ε²/2) × Q_concentric（一階近似）

### Phase 3 通過標準

- [ ] 速度場達到穩態（residual < 1e-8）
- [ ] u(y,z) 與解析解 L₂ 誤差 < 1e-3
- [ ] 最大速度位於 wide gap 側
- [ ] ε→0 極限退化為同心環 Poiseuille
- [ ] 壁面摩擦係數與理論值吻合 < 5%

---

## 附錄 A — 需要 Edit1_GILBM 原始碼確認的項目

以下項目在撰寫本計畫時，因為沒有 solver 原始碼無法精確定位。一旦上傳 `Edit1_GILBM/` 資料夾，可以逐項對照並細化：

| 項目 | 需要看的檔案 | Phase |
|------|------------|-------|
| 截面網格讀取函數 | 讀 .DAT 的 `.cu` / `.h` | 0 |
| metric terms 計算 kernel | Jacobian, g_ij 的 CUDA kernel | 0 |
| i 方向 (spanwise) 週期性 BC | boundary kernel | 1 |
| j 方向 (streamwise) 週期性 + buffer | boundary kernel | 1 |
| force controller (壓力驅動) | body force kernel | 1 |
| k=NK-1 top boundary condition | boundary kernel | 2 |
| GILBM streaming kernel | interpolation + collision | 1, 2 |
| 初始化 kernel | 速度/密度初始化 | 3 |

### 上傳建議

**最小上傳清單：**

```
Edit1_GILBM/
  J_Frohlich/          ← 網格讀取 & metric 計算
  *.cu                 ← 主要 CUDA kernels
  *.h / *.cuh          ← Header files
  main.cu              ← 主程式（確認流程）
  Makefile             ← 確認編譯結構
```

---

## 附錄 B — GILBM 穩定性對照表

| | Periodic Hill | Eccentric Annulus |
|---|---|---|
| GAMMA=0.0 | ω=0.50 OPTIMAL | ω=0.85 OK |
| GAMMA=1.0 | ω=0.55 OPTIMAL | ω=1.20 OK |
| GAMMA=2.0 | ω=0.63 OPTIMAL | ω=1.99 **MARGINAL** |
| GAMMA=2.5 | ω=0.73 OPTIMAL | ω>2.0 **UNSTABLE** |
| 建議 GAMMA | 2.0 | **1.0** |
| 原因 | 穩定窗口寬 | bipolar 度量本身有 8.8x dr_ratio |

---

## 附錄 C — 座標方向快速對照卡

```
┌────────────────────────────────────────────────────────────┐
│  PERIODIC HILL              ECCENTRIC ANNULUS               │
│  ──────────────             ─────────────────               │
│  i → x (span, 均勻)        i → x (stream, 均勻+驅動)       │
│  j → y (stream, 週期)      j → 截面環向 (封閉週期)          │
│  k → z (wall-norm, .DAT)   k → 截面徑向 (.DAT, 雙壁)      │
│                                                             │
│  壁面: k=0 only            壁面: k=0 AND k=NK-1            │
│  驅動: j (y) 方向           驅動: i (x) 方向                │
│  buffer: j 有               buffer: j 無（幾何封閉）         │
│  GAMMA: 2.0 safe           GAMMA: 1.0 safe                 │
└────────────────────────────────────────────────────────────┘
```

---

## 執行順序建議

```
Week 1:  Phase 0 — 網格讀取 + metric 驗證 (Python ↔ CUDA 交叉比對)
Week 2:  Phase 1 — 方向對齊 + 週期性 + 驅動力
Week 3:  Phase 2 — 雙壁面 BC
Week 4:  Phase 3 — 層流驗證 + 解析解比對
```

每個 Phase 結束時做 checkpoint，確認通過標準全部勾選才進入下一步。
