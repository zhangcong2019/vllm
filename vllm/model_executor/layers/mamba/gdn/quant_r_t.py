"""针对正态分布 N(0, σ²) 的 int4 最优量化网格设计。

背景 (见 main.py 的 analyze_uv_distribution):
    SVD 得到的 U/V 元素近似服从 N(0, 1/n) 的正态分布(零均值、对称)。
    均匀 min-max 量化在这种钟形分布上并非最优——它对稀疏的尾部与
    密集的中心一视同仁,浪费码字。信息论上,固定层级数 (int4 = 16 级) 下
    最小化量化 MSE 的标量量化器是 Lloyd-Max 量化器:

      - 判决边界 t_i 取相邻重构点的中点:  t_i = (y_i + y_{i+1}) / 2
      - 重构点 y_i 取所在区间的条件质心:   y_i = E[x | t_{i-1} < x <= t_i]

    对高斯输入,最优网格是"非均匀"的:中心附近层级密集(步长小),
    尾部层级稀疏(步长大),与均匀网格形成对比。

本模块:
    1. lloyd_max_gaussian: 迭代求 N(0,σ²) 的最优 L 级重构点与边界
    2. uniform_grid_gaussian: 对比用的均匀 min-max 网格 (裁剪到 ±k·σ)
    3. quantize_with_levels: 用给定重构点/边界做最近邻量化
    4. main: 对 int4 (16 级) 打印最优网格,并与均匀量化对比 MSE / SQNR
"""

import torch


def lloyd_max_gaussian(
    levels: int = 16,
    sigma: float = 1.0,
    n_samples: int = 2_000_000,
    iters: int = 200,
    seed: int = 0,
    dtype: torch.dtype = torch.float64,
) -> tuple:
    """用 Lloyd-Max 迭代求 N(0, σ²) 的最优 `levels` 级量化器。

    采用大样本蒙特卡洛估计条件质心 (对高斯闭式解涉及误差函数,采样实现更简洁
    且对任意分布通用)。返回 (重构点 y, 内部判决边界 t)。
      - y: (levels,)   升序重构点
      - t: (levels-1,) 内部边界, t_i = (y_i + y_{i+1}) / 2
    """
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n_samples, generator=g, dtype=dtype) * sigma

    # 初始化:按分位数放置重构点,保证每个区间初始都有样本
    qs = (torch.arange(levels, dtype=dtype) + 0.5) / levels
    y = torch.quantile(x, qs)

    for _ in range(iters):
        t = (y[1:] + y[:-1]) / 2                       # (levels-1,) 中点边界
        idx = torch.bucketize(x, t)                    # 每个样本落入的区间 [0, levels-1]
        y_new = y.clone()
        for i in range(levels):
            mask = idx == i
            if mask.any():
                y_new[i] = x[mask].mean()              # 区间条件质心
        if torch.allclose(y_new, y, atol=1e-9):
            y = y_new
            break
        y = y_new

    t = (y[1:] + y[:-1]) / 2
    return y, t


def uniform_grid_gaussian(levels: int = 16, sigma: float = 1.0, clip_sigmas: float = 3.0,
                          dtype: torch.dtype = torch.float64) -> tuple:
    """对比基线:在 [-clip_sigmas·σ, +clip_sigmas·σ] 上的均匀 `levels` 级网格。

    等价于把动态范围裁剪到 ±clip_sigmas·σ 后做均匀 (min-max) 量化。
    返回 (重构点 y, 内部判决边界 t),形式与 lloyd_max_gaussian 一致。
    """
    lo, hi = -clip_sigmas * sigma, clip_sigmas * sigma
    y = torch.linspace(lo, hi, levels, dtype=dtype)
    t = (y[1:] + y[:-1]) / 2
    return y, t


def quantize_with_levels(x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """用重构点 y 与内部边界 t 对 x 做最近邻量化,返回反量化结果。"""
    idx = torch.bucketize(x, t)
    return y[idx]


def companding_gaussian_int4(x: torch.Tensor, levels: int = 16, sigma: float = 1.0) -> torch.Tensor:
    """无查表的高斯最优量化 (压扩/companding),闭式 erf/erfinv 实现。

    非均匀量化 = 非线性压缩 F(x) -> 均匀量化 -> 非线性展开 F^{-1}。
    MSE 最优量化器的高分辨率渐近点密度 ∝ p(x)^{1/3}。对 N(0,σ²),
    p(x)^{1/3} ∝ exp(-x²/(6σ²)),其归一化累积分布正是 erf,故:

        压缩:  u = erf( x / (√6·σ) )              # u ∈ (-1, 1)
        均匀量化: q = round((u+1)/2·L - 0.5),再映射回码字中点
        展开:  x̂ = √6·σ · erfinv( u_deq )

    整个过程只用初等/特殊函数 (erf, erfinv),无需存储重构点/边界表,
    因此可在硬件上直接算 (或用多项式近似 erf)。为高分辨率渐近最优,
    在 L=16 时略逊于精确 Lloyd-Max,但非常接近。
    """
    c = (6.0 ** 0.5) * sigma
    u = torch.erf(x / c)                              # 压缩到 (-1, 1)
    # 均匀量化 u∈[-1,1] 到 L 个中点码字: 码字中心 (2k+1)/L - 1, k=0..L-1
    k = torch.clamp(torch.floor((u + 1.0) / 2.0 * levels), 0, levels - 1)
    u_deq = (2.0 * k + 1.0) / levels - 1.0            # 反量化到码字中点
    return c * torch.erfinv(u_deq)                    # 展开回原尺度


def _mse_sqnr(x: torch.Tensor, xq: torch.Tensor) -> tuple:
    mse = ((x - xq) ** 2).mean().item()
    signal = (x ** 2).mean().item()
    sqnr_db = 10.0 * torch.log10(torch.tensor(signal / mse)).item()
    return mse, sqnr_db


def main() -> None:
    levels = 16          # int4
    sigma = 1.0 / (128 ** 0.5)   # U/V 元素理论标准差 N(0, 1/n), n=128
    print(f"目标分布: N(0, σ²), σ = {sigma:.6e}  (对应 n=128 的 U/V 元素)")
    print(f"量化层级: int4 = {levels} 级\n")

    # 评估用独立测试样本
    g = torch.Generator().manual_seed(1234)
    x_test = torch.randn(2_000_000, generator=g, dtype=torch.float64) * sigma

    # --- Lloyd-Max 最优网格 ---
    y_lm, t_lm = lloyd_max_gaussian(levels=levels, sigma=sigma)
    xq_lm = quantize_with_levels(x_test, y_lm, t_lm)
    mse_lm, sqnr_lm = _mse_sqnr(x_test, xq_lm)

    print("=== Lloyd-Max 最优量化网格 (非均匀) ===")
    print("重构点 y (单位: σ):")
    print("  " + ", ".join(f"{v / sigma:+.4f}" for v in y_lm.tolist()))
    print("重构点 y (原始尺度):")
    print("  " + ", ".join(f"{v:+.4e}" for v in y_lm.tolist()))
    print(f"MSE = {mse_lm:.6e}   SQNR = {sqnr_lm:.3f} dB\n")

    # --- 均匀网格对比 (取几种裁剪范围) ---
    print("=== 均匀 min-max 网格对比 ===")
    best = None
    for k in (2.0, 2.5, 3.0, 4.0):
        y_u, t_u = uniform_grid_gaussian(levels=levels, sigma=sigma, clip_sigmas=k)
        xq_u = quantize_with_levels(x_test, y_u, t_u)
        mse_u, sqnr_u = _mse_sqnr(x_test, xq_u)
        if best is None or mse_u < best[1]:
            best = (k, mse_u, sqnr_u)
        print(f"  裁剪 ±{k}σ: 步长={2*k*sigma/(levels-1):.4e}  MSE={mse_u:.6e}  SQNR={sqnr_u:.3f} dB")
    print(f"  最优均匀裁剪: ±{best[0]}σ  SQNR={best[2]:.3f} dB\n")

    # --- 无查表压扩 (companding) ---
    xq_cp = companding_gaussian_int4(x_test, levels=levels, sigma=sigma)
    mse_cp, sqnr_cp = _mse_sqnr(x_test, xq_cp)
    print("=== 无查表压扩量化 (erf/erfinv, 闭式) ===")
    print(f"MSE = {mse_cp:.6e}   SQNR = {sqnr_cp:.3f} dB")
    print(f"相对 Lloyd-Max 差距: {sqnr_lm - sqnr_cp:.3f} dB\n")

    # --- 总结 ---
    gain = sqnr_lm - best[2]
    print("=== 结论 ===")
    print(f"Lloyd-Max 相对最优均匀量化提升 SQNR ≈ {gain:.3f} dB "
          f"(MSE 降低 {(1 - mse_lm / best[1]) * 100:.1f}%)")
    print("最优网格非均匀:中心步长小、尾部步长大,契合高斯钟形密度。")


def print_optimal_grids(bits_list=(4, 5, 6, 7, 8), sigma: float = 1.0) -> dict:
    """对 int4~int8 分别求标准正态 N(0,1) 的 Lloyd-Max 最优网格并打印 (单位: σ)。

    重构点对 σ 线性缩放,故只需对 N(0,1) 求解一次;实际使用时乘以每列 std。
    返回 {bits: 重构点列表(σ单位)}。
    """
    g = torch.Generator().manual_seed(1234)
    x_test = torch.randn(2_000_000, generator=g, dtype=torch.float64) * sigma

    grids = {}
    for bits in bits_list:
        levels = 2 ** bits
        y, t = lloyd_max_gaussian(levels=levels, sigma=sigma)
        xq = quantize_with_levels(x_test, y, t)
        _, sqnr = _mse_sqnr(x_test, xq)
        y_std = (y / sigma).tolist()
        grids[bits] = y_std
        print(f"\n=== int{bits} ({levels} 级) Lloyd-Max 最优网格  SQNR={sqnr:.3f} dB ===")
        print("重构点 y (单位: σ):")
        # 每行 8 个,便于阅读
        for i in range(0, levels, 8):
            print("  " + ", ".join(f"{v:+.4f}" for v in y_std[i:i + 8]))
    return grids


if __name__ == "__main__":
    main()
    print("\n\n############ int1~int8 最优网格 (标准正态 N(0,1)) ############")
    print_optimal_grids(bits_list=(1, 2, 3, 4, 5, 6, 7, 8))
