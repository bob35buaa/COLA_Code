# 任务计划：检查 COLA 训练环境与数据是否就绪

## 目标
对照 README 与 docs/training.md，判断本机是否具备 Phase 1–3 训练所需的软件环境、仿真依赖、资产与数据。

## 当前阶段
阶段 6

### 阶段 6：补装 isaaclab 核心包
- [ ] 将 flatdict pin 改为 4.1.0
- [ ] pip install -e IsaacLab/source/isaaclab
- [ ] 验证 import isaaclab / AppLauncher
- **状态：** in_progress

## 各阶段

### 阶段 1：对照文档梳理训练前置条件
- [x] 阅读 README、docs/training.md、安装脚本
- [x] 对照论文确认数据/训练设定
- [x] 将发现记录到 findings.md
- **状态：** complete

### 阶段 2：检查软件环境
- [x] conda/python、PyTorch、Isaac Sim、IsaacLab、rsl_rl、项目 editable install
- [x] GPU、驱动、GLIBC
- [x] setup_env.sh 与子模块完整性
- **状态：** complete

### 阶段 3：检查训练数据与资产
- [x] USD/机器人资产
- [x] logs/checkpoints（Phase1/2/3）
- [x] 是否需要外部数据集
- **状态：** complete

### 阶段 4：汇总结论
- [x] 给出「能否开训」的明确结论与缺口清单
- **状态：** complete

### 阶段 5：交付
- [x] 向用户报告环境与数据状态
- **状态：** complete

## 关键问题
1. conda cola 是否完整？**半套：torch/isaacsim/rsl_rl 齐，isaaclab 核心未安装。**
2. 子模块？**已检出。**
3. 外部人类/运动数据？**不需要。**
4. Phase-1/2 checkpoint？**没有。**
5. GPU 是否够 8×2048？**只有 1 张 5090，不能按文档原样跑 Phase 2/3。**

## 已做决策
| 决策 | 理由 |
|------|------|
| 只做只读诊断，不安装/不改环境 | 用户要求「看看当前状态」 |

## 遇到的错误
| 错误 | 尝试次数 | 解决方案 |
|------|---------|---------|
| isaacsim EULA 交互阻塞 | 1 | source setup_env.sh |
| ModuleNotFoundError: isaaclab | 1 | 记录为阻塞项，未擅自重装 |

## 备注
- 结论：软件栈接近 README，但 **还不能训练**。
