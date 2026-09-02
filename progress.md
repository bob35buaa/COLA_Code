# 进度日志

## 会话：2026-09-02（续：补装 isaaclab）

### 阶段 6：补装 isaaclab 核心包
- **状态：** complete
- 执行的操作：
  - IsaacLab/source/isaaclab/setup.py：`flatdict==4.0.1` → `flatdict==4.1.0`
  - `pip install flatdict==4.1.0`
  - `pip install -e IsaacLab/source/isaaclab`
- 测试结果：
  - `import isaaclab` / `AppLauncher` 成功
  - torch 仍为 2.7.0+cu128，CUDA 可用
  - `isaaclab.utils` 在 SimulationApp 启动前会因 `pxr` 报错，这是 Isaac Sim 导入顺序限制，训练脚本先起 AppLauncher 不受影响

## 会话：2026-09-02

### 阶段 1：对照文档梳理训练前置条件
- **状态：** complete
- 执行的操作：
  - 阅读 README、docs/training.md、训练脚本、论文方法部分
- 结论：在线 RL，无离线数据集；Phase 2/3 依赖上一阶段 checkpoint。

### 阶段 2：检查软件环境
- **状态：** complete
- 执行的操作：
  - 检查 conda cola、torch CUDA、isaacsim 5.1、子模块、pip 包
- 结论：isaaclab 核心包未安装，训练入口无法 import。

### 阶段 3：检查训练数据与资产
- **状态：** complete
- 执行的操作：
  - 核对 USD、logs、student.jit、Nucleus 材质引用
- 结论：资产齐；无训练 checkpoint；MuJoCo jit 仅用于 sim2sim。

### 阶段 4–5：汇总结论并交付
- **状态：** complete

## 测试结果
| 测试 | 输入 | 预期结果 | 实际结果 | 状态 |
|------|------|---------|---------|------|
| torch CUDA | cola env | cuda available | RTX 5090, cap 12.0 | pass |
| import isaacsim | EULA=YES | 成功 | 成功 | pass |
| import isaaclab | cola env | 成功 | ModuleNotFoundError | fail |
| import rsl_rl runners | cola env | End2end / WholePipeResi | 成功 | pass |
| logs/model_*.pt | 仓库 | 可衔接 Phase2/3 | 不存在 | fail |
| GPU 数量 | training.md 8 卡 | 8 | 1×5090 | mismatch |

## 错误日志
| 时间戳 | 错误 | 尝试次数 | 解决方案 |
|--------|------|---------|---------|
| 2026-09-02 | isaacsim EULA EOF | 1 | OMNI_KIT_ACCEPT_EULA=YES |
| 2026-09-02 | No module named isaaclab | 1 | 记为环境缺口 |

## 五问重启检查
| 问题 | 答案 |
|------|------|
| 我在哪里？ | 阶段 5 完成 |
| 我要去哪里？ | 等待用户决定是否补装 isaaclab / 是否单卡开 Phase 1 |
| 目标是什么？ | 判断本机是否具备训练环境和数据 |
| 我学到了什么？ | 见 findings.md |
| 我做了什么？ | 只读检查，未改环境 |
