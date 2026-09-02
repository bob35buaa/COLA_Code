# 发现与决策

## 需求
- 对照 README 与 docs/training.md，判断本机是否具备 Phase 1–3 训练所需的软件栈、仿真依赖、资产与数据。

## 研究发现

### 论文与代码的数据形态
- COLA 是 **Isaac Sim 在线强化学习**，不是离线模仿学习。论文三阶段：WBC → residual teacher（闭环仿真人-物-机器人）→ student BC distillation。
- **不需要** AMASS / mocap / 人类轨迹数据集。人的一侧由 `BarController` 在仿真里生成。
- 真正需要的「数据」是：G1 USD 资产、程序化地形、以及 Phase 2/3 衔接用的 **上一阶段 checkpoint**。

### 软件栈（conda env `cola`，2026-09-02 18:33 创建）
| 项 | 期望 | 本机 | 状态 |
|---|---|---|---|
| Python | 3.11 | 3.11.16 | OK |
| PyTorch | 2.7.0 cu128 | 2.7.0+cu128，CUDA 可用 | OK |
| Isaac Sim | 5.1.0 | 5.1.0.0 pip | OK |
| numpy/gymnasium/wandb/tb/mujoco | requirements.txt | 全部匹配 | OK |
| rsl_rl COLA fork | editable | 0.1.0，runners 齐全 | OK |
| cola-code | editable | 0.1.0 | OK |
| IsaacLab 子模块 | 检出 | 2.3.2 @ 37ddf626 | 源码在 |
| **isaaclab 核心包** | `pip install -e IsaacLab/source/isaaclab` | **未注册进 site-packages，`import isaaclab` 失败** | **阻塞** |
| isaaclab 扩展 | assets/tasks/rl/mimic | 已 editable 安装 | 半套 |
| GLIBC | ≥ 2.35 | 2.35 | OK |
| 驱动 | 较新 NVIDIA | 580.159.03 / CUDA 13.0 | OK |

isaaclab 核心缺失的连带依赖也未装：prettytable、hidapi、pyglet、transformers、einops、warp、flatdict。源码目录有 `isaaclab.egg-info`，说明安装曾启动过但未完成/未登记。

### 硬件 vs training.md
- 文档示例：8 GPU × 2048 envs。
- 本机：**1 × RTX 5090 32GB**。当前 sugar 进程占用约 8.8 GB。
- Phase 1 可在单卡上减 `num_envs` 训练。
- Phase 2/3：`train_collaboration.py` 强制 `--distributed`，三群体至少 3 rank；官方拓扑是 8 rank。单卡无法按文档原样开训。

### 资产与 checkpoint
- G1 USD 齐全（含 39MB base mesh 与 fixed-bar usda）。
- 地形为程序生成，不依赖外部 heightmap。
- 场景视觉材质指向 NVIDIA Nucleus（大理石 MDL、HDR 天空），首次启动 Isaac 会联网拉取；不是训练标签数据。
- **没有 `logs/`，没有任何 `model_*.pt`。**
- `deployment/mujoco/student.jit`（14MB，git 跟踪）仅用于 MuJoCo sim2sim，**不能**作为 Phase 2/3 的 `--resume` checkpoint。

### 其他
- 本终端 `DISPLAY` 为空（tty session）。`--headless` 训练不依赖 GUI；有界面评估需要图形会话。
- `pip check` 有 isaacsim 与 torchaudio/click/numpy 等版本告警，属 pip 发行常见噪声，不是当前主阻塞。
- 同时装着上游 `rsl-rl-lib 3.1.2` 与 COLA `rsl_rl 0.1.0`；当前 import 打到 COLA fork。

## 技术决策
| 决策 | 理由 |
|------|------|
| 只读诊断，不重装 | 用户问状态，不要求改环境 |
| 判定「还不能按 training.md 开训」 | isaaclab 核心未安装 + 无 Phase checkpoint + 单卡对不上 8-GPU 拓扑 |

## 遇到的问题
| 问题 | 解决方案 |
|------|---------|
| `import isaacsim` 无 EULA 时卡在交互提示 | `source setup_env.sh` 设置 `OMNI_KIT_ACCEPT_EULA=YES` |
| `import isaaclab` ModuleNotFoundError | 需补装 `IsaacLab/source/isaaclab` 及其依赖 |
| 无 logs/checkpoint | Phase 1 可从零训；Phase 2/3 必须先有上一阶段权重 |

## 资源
- README.md、docs/training.md、setup_env.sh
- 论文 PDF（arXiv 2510.14293）
- conda: `/home/ubuntu/miniconda3/envs/cola`

## 视觉/浏览器发现
- 未开浏览器。论文确认训练为仿真闭环 RL，无外部人类数据集。
