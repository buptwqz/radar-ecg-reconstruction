# radar-ecg-reconstruction

本仓库包含两部分：

1. **雷达信号处理 / 周期信号提取（Notebook）**：用于从数据集中提取相位/周期相关信号，做数据理解与预处理验证。
2. **雷达到 ECG 重建（条件扩散模型 + 实时/离线推理工具）**：位于 `diffu_1228/radar-ecg-diffusion/`，包含训练、评估、离线推理（文件/批处理/监控）以及 IWR1443 实时 GUI。

> 如果你主要想训练模型/跑推理/开 GUI：请直接从 `diffu_1228/radar-ecg-diffusion/` 开始。

---

## 目录结构（概览）

```
radar-ecg-reconstruction/
├── dacm_extract_phase.ipynb                  # 信号提取/分析 Notebook
├── diffu_1228/
│   └── radar-ecg-diffusion/                  # 扩散模型子项目（训练/评估/推理/GUI）
│       ├── dataset_*.npz                     # 示例数据集（npz，radar/ecg）
│       ├── scripts/                          # 训练/评估脚本
│       ├── src/                              # 模型、推理、GUI、数据解析
│       ├── checkpoints/                      # 训练输出/示例权重
│       └── configs/default.yaml              # 默认配置（部分脚本参考）
└── README.md
```

---

## 环境准备（Windows / 推荐）

建议使用 Python 3.10+（或 3.9+），并为本项目创建独立虚拟环境。

进入扩散模型子项目目录：

```powershell
cd diffu_1228\radar-ecg-diffusion
```

安装基础依赖（训练/评估/采样）：

```powershell
pip install -r requirements.txt
```

如果你需要运行 **GUI / 串口 / UDP / HTTP 上报 / 文件监控**，还需要额外依赖：

```powershell
pip install PyQt6 pyqtgraph pyserial requests watchdog
```

> 说明：`requirements.txt` 目前偏向训练/评估；GUI/串口/HTTP/监控工具依赖在 `src/*.py` 中使用。

---

## 数据格式

训练与评估脚本默认使用 `.npz` 数据集，内部包含两个数组：

- `radar`: 形状 `(N, L)`
- `ecg`: 形状 `(N, L)`

其中 `L` 必须与训练/推理输入长度一致（仓库内主要使用 **128 点**；也提供 512 点数据集与准备脚本）。

---

## 快速开始：训练 / 评估（128 点）

在 `diffu_1228/radar-ecg-diffusion/` 下执行。

### 1) 训练

```powershell
python scripts\train.py --data dataset_128_0113.npz --length 128 --epochs 50 --batch-size 32 --lr 2e-4 --tag 128_0113
```

训练完成后会输出权重到：

```
checkpoints/diffusion_model_final_<tag>.pth
```

### 2) 评估（并绘图）

```powershell
python scripts\evaluate.py --data dataset_128_0113.npz --model checkpoints\diffusion_model_final_128_0113.pth --ddim-steps 20
```

---

## 数据集准备（512 点示例）

子项目提供脚本将“雷达序列文件夹 + ECG 序列文件夹”整合为 `.npz`：

```powershell
python prepare_dataset_512.py --radar-dir ..\..\radar-data-0113 --ecg-dir ..\..\ecg-data-0113 --output dataset_512_0113.npz --length 512
```

如果要用 512 点训练，需要保证训练/推理入口中的 `--length 512` 与模型/数据一致（当前部分脚本与 GUI 默认按 128 点设计）。

---

## 离线推理（文件/批处理/目录监控）

适合你已经将雷达序列落盘为 CSV（每个文件是一段 1D 序列），不需要连雷达硬件。

脚本：`diffu_1228/radar-ecg-diffusion/src/ecg_diffusion_file_monitor.py`

### 单文件推理

```powershell
python src\ecg_diffusion_file_monitor.py --mode single --input path\to\radar.csv --model checkpoints\diffusion_model_final_128_0113.pth --ddim-steps 50
```

### 批量推理（目录内所有 CSV）

```powershell
python src\ecg_diffusion_file_monitor.py --mode batch --input path\to\radar_dir --output inference_output --model checkpoints\diffusion_model_final_128_0113.pth
```

### 监控目录（有新 CSV 就自动推理）

```powershell
python src\ecg_diffusion_file_monitor.py --mode realtime --input path\to\watch_dir --output inference_output --model checkpoints\diffusion_model_final_128_0113.pth
```

---

## 实时推理 GUI（IWR1443 串口）

脚本：`diffu_1228/radar-ecg-diffusion/src/iwr1443_realtime_ecg_diffusion_gui.py`

功能：

- 通过 **CLI 串口**下发 cfg
- 从 **Data 串口**读取 vitalSigns 输出（解析结构体字段）
- 以滑窗方式触发扩散模型推理（默认 128 点输入，步长可在 GUI 调整）
- 实时显示重建 ECG，并基于重建 ECG 做简易心率估计

示例启动：

```powershell
python src\iwr1443_realtime_ecg_diffusion_gui.py --data-port COM14 --cli-port COM13 --cfg src\profile_2d_VitalSigns.cfg --model src\checkpoints\diffusion_model_final_128_0122.pth --ddim-steps 20
```

> 注意：默认参数里包含作者机器的绝对路径；建议启动时显式传入 `--cfg` 和 `--model`，并按你的 COM 口修改。

---

## 其他 GUI / 上报工具（可选）

子项目 `src/` 下还包含一些变体脚本（按你的链路选择）：

- `iwr1443_serial_vitals_gui.py`：串口版本生命体征 GUI
- `iwr1443_udp_http_vitals_gui.py`：UDP 接收雷达数据 + HTTP 上报
- `iwr1443_udp_http_vitals_ecg_gui.py`：UDP + 扩散 ECG 重建 + HTTP 逐点上报
- `iwr1443_udp_http_vitals_ecg_sync_sender.py`：整段同步发送（将 breath/heart/ecg 归一化后批量上报）

---

## Notebook：信号提取/分析

根目录的 `dacm_extract_phase.ipynb` 用于雷达信号处理与周期信号提取（例如相位相关分析、重采样/滤波等探索）。

建议使用 Jupyter（VS Code Notebook 也可）打开运行。

---

## 常见问题（FAQ）

1. **训练时报“数据长度不匹配”**
	- `scripts/train.py` 会强制检查 `L`，请确保 `.npz` 中的 `radar/ecg` 形状为 `(N, 128)`，并传入 `--length 128`。

2. **GUI 打不开 / 缺少依赖**
	- 安装：`pip install PyQt6 pyqtgraph pyserial requests watchdog`。

3. **CUDA 不可用**
	- 训练/推理会自动回落到 CPU，但速度会明显变慢；可先验证流程，再切换 GPU。

---

## 子项目文档

更偏“模型工程”的说明见：`diffu_1228/radar-ecg-diffusion/README.md`。
