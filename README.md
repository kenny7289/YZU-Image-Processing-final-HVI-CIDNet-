# HVI-CIDNet Demo（個人研究實驗環境）

> 本 Demo 環境是使用 **Claude Code**（Anthropic 推出的 VS Code 擴充功能）參考官方 repo
> [Fediory/HVI-CIDNet](https://github.com/Fediory/HVI-CIDNet) 所建置與修改而成，
> 用於個人研究論文實驗，非官方原始 repo。

官方原始論文與完整說明請參閱 [ORIGINAL_README.md](ORIGINAL_README.md)（官方 repo 原始 README，
因 Windows 檔案系統不分大小寫，無法與本檔案同時取名為 `Readme.md` / `README.md`，故改名保留）。

## 環境建置摘要

本環境的建置步驟如下（已自動完成）：

1. `git clone https://github.com/Fediory/HVI-CIDNet.git`
2. 安裝 Miniconda3，並建立 `python=3.7` 的 conda 環境 `hvi-cidnet`
3. 安裝 `requirements.txt` 中的相依套件
   - 註：`safetensors` 限制為 `<0.4`，因為 0.4 以上版本在 Windows + Python 3.7 沒有預編譯
     wheel，需要 Rust 工具鏈才能從原始碼建置，會導致安裝失敗。
   - 註：`scikit-image` 固定為 `==0.18.3`，因為新版（0.19+）的 `rgb2gray` 對已是灰階的
     2D 陣列會直接報錯，導致 `image_quality`（BRISQUE）套件在「Image Score = Yes」時崩潰。
4. 從 Hugging Face 模型庫 [`Fediory/HVI-CIDNet-Generalization`](https://huggingface.co/Fediory/HVI-CIDNet-Generalization)
   下載官方推薦的泛化（generalization）模型權重，轉換並存放於
   `weights/LOLv2_syn/generalization.pth`
5. 修改 [app.py](app.py)，在標題與頁面底部加上 Claude Code 建置聲明，並補上簡單使用說明

## 安裝環境（重新建置時參考）

```bash
# 1. 建立並啟用 conda 環境
conda create -n hvi-cidnet python=3.7 -y
conda activate hvi-cidnet

# 2. 安裝相依套件
pip install -r requirements.txt

# 3. 下載 generalization 權重（會存到 weights/LOLv2_syn/generalization.pth）
python download_generalization.py
```

## 執行 Demo

```bash
conda activate hvi-cidnet

# 有 GPU（CUDA）
python app.py

# 沒有 GPU，純 CPU 推論
python app.py --cpu
```

啟動後，瀏覽器開啟 <http://127.0.0.1:7862> 即可使用。**請用獨立的瀏覽器視窗（Chrome / Edge /
Firefox）開啟，不要用 VS Code 內建的 Simple Browser**，否則 WebSocket 連線不完整會導致畫面一直顯示
Error。

## 使用說明

1. 上傳一張低光照（low-light）圖片。
2. 在「Model Weights」選擇權重檔，建議使用 `generalization.pth`（泛化能力最佳，已預先下載好）。
3. 調整 `gamma curve`、`Alpha-s`（飽和度）、`Alpha-i`（亮度）滑桿來控制增強效果。
4. 「Image Score」選 `Yes` 可額外計算 NIQE / BRISQUE 影像品質指標（計算較耗時，預設 `No`）。
5. 沒有 GPU 時，啟動加上 `--cpu` 參數即可在 CPU 上執行。

## 目錄結構（與官方 repo 的差異）

- `app.py`：在 `gr.Interface` 加上 `title` 後綴與 `article`（頁尾）的 Claude Code 建置聲明，
  並新增 `description` 區塊作為簡單使用說明。
- `requirements.txt`：將 `safetensors` 限制為 `<0.4`、`scikit-image` 固定為 `==0.18.3`，
  以相容 Windows + Python 3.7 並避免 BRISQUE 計算崩潰。
- `download_generalization.py`：新增的小工具腳本，從 Hugging Face 下載官方泛化模型權重並轉成
  `.pth` 格式存放於 `weights/LOLv2_syn/generalization.pth`。
- `weights/LOLv2_syn/generalization.pth`：官方推薦的泛化模型權重（由 Hugging Face 下載）。
- `ORIGINAL_README.md`：官方原始 README（由 `Readme.md` 改名保留）。

其餘程式碼（`net/`、`loss/`、`eval.py`、`train.py` 等）皆未更動，維持官方原始實作。

---

# HVI-CIDNet Demo (Personal Research Experiment Environment)

> This demo environment was built and modified with **Claude Code** (Anthropic's VS Code
> extension), referencing the official repo
> [Fediory/HVI-CIDNet](https://github.com/Fediory/HVI-CIDNet), for personal research paper
> experiments. This is not the official original repo.

For the original paper and full documentation, see [ORIGINAL_README.md](ORIGINAL_README.md)
(the official repo's original README — kept under this name because Windows' case-insensitive
filesystem won't allow `Readme.md` and `README.md` to coexist as separate files).

## Setup Summary

The environment was built with the following steps (already done automatically):

1. `git clone https://github.com/Fediory/HVI-CIDNet.git`
2. Installed Miniconda3 and created a `python=3.7` conda environment named `hvi-cidnet`
3. Installed the dependencies in `requirements.txt`
   - Note: `safetensors` is pinned to `<0.4`, because versions 0.4+ have no prebuilt wheel for
     Windows + Python 3.7 and require a Rust toolchain to build from source, which would fail.
   - Note: `scikit-image` is pinned to `==0.18.3`, because newer versions (0.19+) make `rgb2gray`
     raise an error on already-grayscale 2D arrays, which crashes the `image_quality` (BRISQUE)
     package when "Image Score = Yes" is selected.
4. Downloaded the officially recommended generalization model weights from the Hugging Face
   model repo [`Fediory/HVI-CIDNet-Generalization`](https://huggingface.co/Fediory/HVI-CIDNet-Generalization),
   converted, and saved to `weights/LOLv2_syn/generalization.pth`
5. Modified [app.py](app.py) to add a Claude Code attribution notice to the title and page
   footer, plus a short usage guide

## Installation (reference for rebuilding the environment)

```bash
# 1. Create and activate the conda environment
conda create -n hvi-cidnet python=3.7 -y
conda activate hvi-cidnet

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download the generalization weights (saved to weights/LOLv2_syn/generalization.pth)
python download_generalization.py
```

## Running the Demo

```bash
conda activate hvi-cidnet

# With GPU (CUDA)
python app.py

# CPU-only, no GPU
python app.py --cpu
```

Once started, open <http://127.0.0.1:7862> in your browser. **Use a standalone browser window
(Chrome / Edge / Firefox), not VS Code's built-in Simple Browser** — its incomplete WebSocket
support causes every request to fail with an "Error" badge.

## Usage Guide

1. Upload a low-light image.
2. Choose a weight file under "Model Weights" — `generalization.pth` is recommended (best
   generalization ability, already downloaded for you).
3. Adjust the `gamma curve`, `Alpha-s` (saturation), and `Alpha-i` (brightness) sliders to
   control the enhancement.
4. Set "Image Score" to `Yes` to additionally compute NIQE / BRISQUE image quality metrics
   (slower; default is `No`).
5. If you don't have a GPU, add the `--cpu` flag at startup to run on CPU.

## Directory Structure (differences from the official repo)

- `app.py`: Added a Claude Code attribution notice appended to the `title` and as an `article`
  (page footer) in `gr.Interface`, plus a new `description` block as a short usage guide.
- `requirements.txt`: Pinned `safetensors` to `<0.4` and `scikit-image` to `==0.18.3` for
  Windows + Python 3.7 compatibility and to avoid the BRISQUE crash.
- `download_generalization.py`: A new helper script that downloads the official generalization
  model weights from Hugging Face and converts them to `.pth` format at
  `weights/LOLv2_syn/generalization.pth`.
- `weights/LOLv2_syn/generalization.pth`: The officially recommended generalization model
  weights (downloaded from Hugging Face).
- `ORIGINAL_README.md`: The official repo's original README (renamed from `Readme.md`).

All other code (`net/`, `loss/`, `eval.py`, `train.py`, etc.) is unmodified and remains the
original official implementation.
