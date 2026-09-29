# FTLE/LCS HPC 项目简历材料

## 项目名称

**FTLE/LCS 高性能计算系统｜CUDA、粒子计算、Multi-GPU、NVLink**

## 简历精简版

- 将 Double-Gyre FTLE 粒子求解器从 CPU reference 重构为 CUDA 实现，通过 multi-step kernel fusion、RK4 时间系数预计算和 `sincospi`/受控混合精度优化，在 $2001\times1001$、1000 步工况上相对 CUDA V1 实现 **3.10×** 加速，达到 **20.84 billion particle-steps/s**；相对 CPU reference 的早期对照达到百倍量级加速。
- 建立 strict FP64 与 mixed-trig 两条数值路径及自动误差门槛，使用 Nsight Systems/Compute 分析 kernel launch、指令数、寄存器和 occupancy；将主 kernel 从 96 registers/thread、40 B local memory 优化到 72 registers/thread、0 B local memory，并验证 mixed 模式在 $\tau=10$ 下 relative-L2 误差为 **1.71e-7**。
- 基于多进程单卡一 CUDA context 的 y-domain decomposition 实现 2×A800 强扩展，取得 **1.948× speedup / 97.4% parallel efficiency**；进一步实现 CUDA IPC mapped P2P 的一行 Halo 交换，在 NV8 上测得 **187.27 GB/s** 峰值单向带宽，并将 Halo 延迟由 host staging 的 **508 μs 降至 33 μs（15.4×）**。

## 简历稳妥版

如果简历空间较小，推荐使用下面三条：

- Optimized a CUDA Double-Gyre FTLE solver with multi-step fusion, coefficient precomputation, paired `sincospi`, and controlled mixed-precision trigonometry; achieved **3.10×** speedup over the CUDA V1 baseline and **20.84B particle-steps/s** on an A100.
- Built strict-FP64 correctness gates and Nsight Systems/Compute profiling workflow; reduced the hot kernel from **96 registers + 40 B local memory/thread** to **72 registers + 0 B**, while keeping relative-L2 FTLE error at **1.71e-7** for the validated $\tau=10$ configuration.
- Implemented process-per-GPU domain decomposition on 2×A800 with **1.948× strong-scaling speedup (97.4% efficiency)**; added CUDA IPC/NVLink halo exchange, reaching **187.27 GB/s** P2P bandwidth and reducing halo latency **15.4×** versus host staging.

## 一分钟面试叙述

这个项目不是简单把 NumPy 代码搬到 GPU。我先建立 CPU 和 strict-FP64 CUDA reference，再逐项做受控优化：减少 RK4 kernel launch、预计算所有时间系数、使用成对 `sincospi` 降低寄存器和 local-memory spill，最后只对三角函数使用 FP32，轨迹、变形梯度和 RK4 累加仍保持 FP64。每次优化都记录 CUDA Event 中位数，并通过 relative-L2、最大绝对误差、边界误差和 $\det(F)$ 不变量验证。

多 GPU 部分有两条路径。direct-F 形式中粒子独立，因此按 y 行分区后可以取得接近线性的 2 GPU 强扩展。为了研究真实通信，我又实现了基于 flow-map 有限差分的 FTLE 路径，它需要交换一行 Halo；每张 GPU 使用独立进程和 CUDA context，通过 CUDA IPC 映射对端显存并执行 P2P 拷贝。结果显示同步 IPC 是当前问题的最佳方案；异步 overlap 反而更慢，因为 32 KB Halo 只有约 30 μs，拆分 interior/boundary kernel 的启动和同步开销超过了可隐藏的通信。

## 可以明确声称的结果

- 单 GPU：A100 实测；最大网格 mixed 相对 CUDA V1 为 3.097×。
- 多 GPU：2×A800 实测；1→2 GPU speedup 为 1.948×，效率 97.4%。
- 互联：A800 NV8 实测；大消息单向 P2P 带宽最高 187.27 GB/s。
- Halo：32,016 bytes/rank；IPC sync 相对 host staging 延迟降低 15.4×。
- 数值：分布式 flow-map 相对 monolithic flow-map relative-L2 为 3.34e-17。

## 不应声称的内容

- 不要写“在 4 GPU 上实现或验证”，因为当前实测只有 2 GPU。
- 不要把 2 GPU 结果线性外推成 4 GPU 数字。
- 不要说 overlap 一定更快；本项目的实测结论恰好是同步 IPC 更合适。
- 不要说 flow-map 与 direct-F 在 $\tau=10$ 下完全一致；两者 relative-L2 差异约为 8.09%，这是有限差分离散误差，不是通信错误。
- 不要把 mixed 模式推广到所有积分窗口；$\tau=20$ 已超过当前最大绝对误差门槛，新参数必须重新验证。

## 项目关键词

CUDA, Numba CUDA, RK4, FTLE/LCS, particle integration, variational equation, kernel fusion, mixed precision, CUDA IPC, P2P, NVLink, domain decomposition, halo exchange, strong scaling, Nsight Systems, Nsight Compute, numerical verification
