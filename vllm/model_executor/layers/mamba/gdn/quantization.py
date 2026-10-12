"""低秩矩阵构造、SVD 分解与重构误差验证 (PyTorch 版本)。"""

import argparse
import re
import sys
import time
from importlib import import_module
from pathlib import Path

import torch

# sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "QuaRot" / "fake_quant"))
# quantize_weight = import_module("gptq_matrix_example").quantize_weight

from .quant_r_t import lloyd_max_gaussian

# 默认 ssm_state dump 根目录 (含 language_model.model.layers.{L}.linear_attn/step_XXXX_cache.pt)
DEFAULT_DUMP_ROOT = "/home/cong/repo/hybrid_cache/cache_dumps_server_9B_you_1547"
_STEP_CACHE_RE = re.compile(r"step_(\d+)_cache\.pt$")


def _layer_dir(dump_root: Path, layer: int) -> Path:
    return dump_root / f"language_model.model.layers.{layer}.linear_attn"


def _resolve_step_file(layer_dir: Path, step: int | None) -> Path:
    """返回指定 step 的 dump 文件;step 为 None 时取最后一个 (step 最大)。"""
    if not layer_dir.is_dir():
        raise SystemExit(f"层目录不存在: {layer_dir}")

    if step is not None:
        step_file = layer_dir / f"step_{step:04d}_cache.pt"
        if not step_file.is_file():
            raise SystemExit(f"step 文件不存在: {step_file}")
        return step_file

    step_files = sorted(
        (f for f in layer_dir.glob("step_*_cache.pt") if _STEP_CACHE_RE.search(f.name)),
        key=lambda f: int(_STEP_CACHE_RE.search(f.name).group(1)),
    )
    if not step_files:
        raise SystemExit(f"目录下无 step_XXXX_cache.pt 文件: {layer_dir}")
    return step_files[-1]


def load_matrix_from_dump(
    dump_root: str, layer: int, head: int, step: int | None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """从 dump 中按 layer/head/step 读取单个 head 的 ssm_state 矩阵 (2D)。"""
    layer_dir = _layer_dir(Path(dump_root), layer)
    step_file = _resolve_step_file(layer_dir, step)

    try:
        data = torch.load(step_file, map_location="cpu", weights_only=False)
    except TypeError:
        data = torch.load(step_file, map_location="cpu")

    if not isinstance(data, dict) or "ssm_state" not in data:
        raise SystemExit(f"dump 不含 'ssm_state': {step_file}")

    ssm_state = data["ssm_state"]
    if ssm_state.ndim < 4:
        raise SystemExit(f"期望 ssm_state 至少 4 维,得到 shape={list(ssm_state.shape)}")
    if not (0 <= head < ssm_state.shape[1]):
        raise SystemExit(f"head={head} 超出范围,shape={list(ssm_state.shape)}")

    selected = ssm_state[0, head]
    print(f"读取 dump: {step_file}")
    print(f"  ssm_state shape={list(ssm_state.shape)} -> selected head={head} shape={list(selected.shape)}")
    return selected.to(dtype)

# 缓存不同 bit 数的 Lloyd-Max 标准正态 N(0,1) 最优重构点 (单位: σ),避免重复迭代求解
_LLOYD_MAX_GRID_CACHE: dict = {}


def _lloyd_max_grid_std(bits: int, dtype: torch.dtype, device) -> torch.Tensor:
    """返回 N(0,1) 上 int`bits` (2^bits 级) 的 Lloyd-Max 最优重构点 (升序, 单位 σ)。"""
    key = (bits, dtype, str(device))
    if key not in _LLOYD_MAX_GRID_CACHE:
        y, _ = lloyd_max_gaussian(levels=2 ** bits, sigma=1.0)
        _LLOYD_MAX_GRID_CACHE[key] = y.to(dtype=dtype, device=device)
    return _LLOYD_MAX_GRID_CACHE[key]


def build_low_rank_matrix(
    n: int = 128,
    r: int = 16,
    seed: int = 7,
    decay_low: float = 0.75,
    decay_high: float = 1.0,
    dtype: torch.dtype = torch.float64,
    device: str = "cpu",
) -> torch.Tensor:
    """构造一个秩为 r 的 n x n 矩阵,其非零奇异值按随机衰减因子递减。

    构造方式: A = U @ diag(s) @ Vt
      - U (n x r)、V (n x r) 为随机正交矩阵(由随机高斯矩阵的 QR 分解得到)
      - s[0] = 1.0, s[i] = s[i-1] * factor[i-1], factor ~ U(decay_low, decay_high)
    这样可精确控制奇异值谱形,便于模拟真实数据中能量集中的低秩结构。
    """
    g = torch.Generator(device=device).manual_seed(seed)

    # 递减奇异值序列
    factors = torch.rand(r - 1, generator=g, dtype=dtype, device=device) * (decay_high - decay_low) + decay_low
    s = torch.empty(r, dtype=dtype, device=device)
    s[0] = torch.ones((), dtype=dtype, device=device)
    s[1:] = torch.cumprod(factors, dim=0)

    # 随机正交基:对 (n, r) 高斯矩阵做 reduced QR,得到 (n, r) 列正交 Q
    U_mat, _ = torch.linalg.qr(torch.randn(n, r, generator=g, dtype=dtype, device=device), mode="reduced")
    V_mat, _ = torch.linalg.qr(torch.randn(n, r, generator=g, dtype=dtype, device=device), mode="reduced")

    A = U_mat @ torch.diag(s) @ V_mat.T
    return A


def energy_rank(s: torch.Tensor, thresholds=(0.90, 0.95, 0.99, 0.999)) -> dict:
    """打印保留各能量阈值 (∑σ²) 所需的最小 rank,返回阈值->rank 字典。"""
    energy = s ** 2
    cum = torch.cumsum(energy, dim=0) / energy.sum()
    result = {}
    print("\n=== 能量 (∑σ²) 累计所需 rank ===")
    for thr in thresholds:
        rank = int(torch.searchsorted(cum, torch.tensor(thr, dtype=cum.dtype, device=cum.device)).item()) + 1
        result[thr] = rank
        print(f"  {thr * 100:.1f}% energy -> rank {rank}")
    return result


def analyze_uv_distribution(U: torch.Tensor, Vt: torch.Tensor, r: int,
                            use_hadamard: bool = True) -> dict:
    """分析 U、V 前 r 列元素的分布,检验是否符合正态分布,打印统计量。

    Haar 随机正交矩阵的元素理论上近似 N(0, 1/n),这里用矩统计
    (均值/标准差/偏度/峰度) 与正态性检验 (Shapiro-Wilk, KS) 验证。
    偏度接近 0、超额峰度接近 0、检验 p 值较大 (>0.05) 则支持正态假设。

    use_hadamard=True 时,先对 U、V 各列左乘归一化 Walsh-Hadamard 矩阵
    (H @ M,沿 n 维旋转) 再分析。正交旋转使能量在各维重新分布、峰值降低,
    分布通常更接近正态,可对比旋转前后的正态性。
    """
    from scipy import stats

    result = {}
    tag = " [Hadamard 旋转后]" if use_hadamard else ""
    print(f"\n=== U/V 元素分布正态性分析 (前 r={r} 列){tag} ===")

    H = None
    if use_hadamard:
        n = U.shape[0]
        H = hadamard_matrix(n, dtype=torch.float64, device="cpu")

    for name, M in [("U", U[:, :r]), ("V", Vt[:r, :].T)]:
        M = M.to(torch.float64).cpu()
        if H is not None:
            M = H @ M                                    # 沿 n 维 Hadamard 旋转
        x = M.flatten().numpy()
        mean, std = x.mean(), x.std()
        skew, kurt = stats.skew(x), stats.kurtosis(x)   # kurtosis 为超额峰度
        sh_w, sh_p = stats.shapiro(x)
        ks_p = stats.kstest((x - mean) / std, "norm").pvalue
        result[name] = {
            "mean": mean, "std": std, "skew": skew, "kurt": kurt,
            "shapiro_W": sh_w, "shapiro_p": sh_p, "ks_p": ks_p,
        }
        normal = "符合正态" if sh_p > 0.05 else "偏离正态"
        print(f"  {name}: n={x.size}  mean={mean:.4e}  std={std:.4e} (理论 {1/(M.shape[0]**0.5):.4e})")
        print(f"     偏度={skew:.4f}  超额峰度={kurt:.4f}")
        print(f"     Shapiro W={sh_w:.4f} p={sh_p:.4e}  KS p={ks_p:.4e}  -> {normal}")
    return result


def reconstruct_with_top_r(
    U: torch.Tensor, s: torch.Tensor, Vt: torch.Tensor, r: int
) -> torch.Tensor:
    """使用前 r 个奇异值重构矩阵。"""
    return U[:, :r] @ torch.diag(s[:r]) @ Vt[:r, :]


def print_quantization_bitrate(
    A: torch.Tensor, U: torch.Tensor, s: torch.Tensor, Vt: torch.Tensor, r: int,
    bits_U: int | torch.Tensor | None, bits_V: int | torch.Tensor | None,
    mxfp4: bool = False,
    gptq: bool = False,
    kmeans: str | None = None,
    per_col: str | None = "scale_zero",
) -> None:
    """打印最终因子的理论码值位宽及含 s 的 bit/A元素,不代表浮点模拟的内存占用。

    bits 为 None 表示保留原 dtype。普通整数按列 scale 为 float32,zero 为 uint8;
    FP4 scale/center、压扩 min/max 按因子 dtype 存储。
    bit 向量按其 dtype 存储,U/V 相同的向量仅计一次。
    固定算法、维度、标量位宽和配置视为双方约定,不计文件头及字节对齐。
    MXFP4 按每列完整 32 元素块计入尾块填充和 8-bit E8M0 scale。
    GPTQ 按 groupsize=-1、每行一组 float32 scale/zero 计入元数据,零点不压缩。
    kmeans 为 per_col 或 per_tensor 时计入各因子原 dtype 的完整码本;
    索引按配置位宽计费,不额外压缩未使用的码字。
    """
    U_r = U[:, :r]
    V_r = Vt[:r, :].T
    width_U = U.element_size() * 8 if bits_U is None else (
        bits_U.to(torch.float64).mean().item() if isinstance(bits_U, torch.Tensor) else bits_U
    )
    width_V = Vt.element_size() * 8 if bits_V is None else (
        bits_V.to(torch.float64).mean().item() if isinstance(bits_V, torch.Tensor) else bits_V
    )
    factor_elements = U_r.numel() + V_r.numel()
    factor_bits = U_r.numel() * width_U + V_r.numel() * width_V
    print(f"U/V 码值位宽:        U={width_U:.4f}, V={width_V:.4f}, 平均={factor_bits / factor_elements:.4f} bit/元素")
    if mxfp4:
        blocks = ((U_r.shape[0] + 31) // 32) * U_r.shape[1]
        blocks += ((V_r.shape[0] + 31) // 32) * V_r.shape[1]
        factor_bits = blocks * (32 * 4 + 8)
        print(f"MXFP4 有效位宽:     {factor_bits / factor_elements:.4f} bit/元素 (含 E8M0 scale 和尾块填充)")
    elif gptq:
        parameter_count = U_r.shape[0] + V_r.shape[0]
        scale_bits = parameter_count * 32
        zero_bits = parameter_count * 32
        factor_bits += scale_bits + zero_bits
        print(f"GPTQ 按行元数据:    scale={parameter_count} x 32 = {scale_bits} bit, zero={parameter_count} x 32 = {zero_bits} bit")
        print(f"GPTQ 有效位宽:      {factor_bits / factor_elements:.4f} bit/元素 (含 float32 scale/zero, 零点未压缩)")
    elif kmeans is not None:
        if kmeans not in ("per_col", "per_tensor"):
            raise ValueError("kmeans must be per_col or per_tensor")
        codebook_bits = 0
        for name, factor, bits in (("U", U_r, bits_U), ("V", V_r, bits_V)):
            if not isinstance(bits, int) or bits < 1:
                raise ValueError("kmeans requires a positive integer bit width")
            if kmeans == "per_col":
                centers = factor.shape[1] * min(2 ** bits, factor.shape[0])
            else:
                centers = min(2 ** bits, factor.numel())
            storage_bits = factor.element_size() * 8
            codebook_bits += centers * storage_bits
            print(f"k-means {name} 码本:    {centers} 个中心 x {storage_bits} bit = {centers * storage_bits} bit ({kmeans})")
        factor_bits += codebook_bits
        print(f"k-means 有效位宽:   {factor_bits / factor_elements:.4f} bit/元素 (含码本)")
    elif per_col is not None:
        parameter_labels = {
            "scale_zero": "scale/zero",
            "scale_center": "scale/center",
            "min_max": "min/max",
        }
        if per_col not in parameter_labels:
            raise ValueError("unknown per-column metadata format")
        for name, factor, bits in (("U", U_r, bits_U), ("V", V_r, bits_V)):
            if bits is None:
                continue
            storage_bits = factor.element_size() * 8
            if per_col == "scale_zero":
                parameter_bits = factor.shape[1] * (32 + 8)
                layout = "32 (float32 scale) + 8 (uint8 zero)"
            else:
                parameter_bits = factor.shape[1] * 2 * storage_bits
                layout = f"2 x {storage_bits}"
            factor_bits += parameter_bits
            print(f"{name} 按列元数据:      {parameter_labels[per_col]}, {factor.shape[1]} x ({layout}) = {parameter_bits} bit")

    allocation_bits = 0
    if isinstance(bits_U, torch.Tensor):
        allocation_bits += bits_U.numel() * bits_U.element_size() * 8
    if isinstance(bits_V, torch.Tensor):
        shared = (isinstance(bits_U, torch.Tensor) and bits_U.dtype == bits_V.dtype
                  and torch.equal(bits_U.reshape(-1), bits_V.reshape(-1)))
        if not shared:
            allocation_bits += bits_V.numel() * bits_V.element_size() * 8
    if allocation_bits:
        print(f"bit 分配元数据:     {allocation_bits} bit (原 dtype,相同 U/V 向量共用一份)")
    factor_bits += allocation_bits
    singular_bits = s[:r].numel() * s.element_size() * 8
    total_bits = factor_bits + singular_bits
    print(f"U/V 含元数据位宽:   {factor_bits / factor_elements:.4f} bit/元素")
    print(f"理论总数据量:        {total_bits:.0f} bit (U/V及元数据={factor_bits:.0f}, s={singular_bits})")
    print(f"理论量化比特率:      {total_bits / A.numel():.6f} bit/A元素 (含解码数据;固定算法/形状/配置共享,不含文件头/对齐)")


def baseline_reconstruction_error(
    A: torch.Tensor, U: torch.Tensor, s: torch.Tensor, Vt: torch.Tensor, r: int
) -> dict:
    """打印 SVD 谱信息与基线 (无损 top-r) 重构误差。"""
    print(f"SVD 完成: U{tuple(U.shape)}, s{tuple(s.shape)}, Vt{tuple(Vt.shape)}")
    print(f"非零奇异值 (前 r={r} 个):")
    for i in range(r):
        if i + 1 < r and s[i] > 0:
            ratio = (s[i + 1] / s[i]).item()
            print(f"  s[{i:2d}] = {s[i].item():.6e}  (s[{i+1}]/s[{i}] = {ratio:.4f})")
        else:
            print(f"  s[{i:2d}] = {s[i].item():.6e}")
    print(f"尾部奇异值 (第 r+1 起近似 0): {s[r:r + 3].tolist()}")

    t0 = time.perf_counter()
    A_r = reconstruct_with_top_r(U, s, Vt, r=r)
    elapsed = time.perf_counter() - t0
    abs_err = torch.linalg.norm(A - A_r, ord="fro").item()
    rel_err = abs_err / torch.linalg.norm(A, ord="fro").item()
    max_err = (A - A_r).abs().max().item()

    print("\n=== 基线重构误差 (无损 top-r) ===")
    print(f"绝对误差 (Frobenius): {abs_err:.6e}")
    print(f"相对误差:            {rel_err:.6e}")
    print(f"最大元素误差:        {max_err:.6e}")
    print(f"重构用时:            {elapsed * 1e3:.3f} ms")
    print_quantization_bitrate(A, U, s, Vt, r, None, None)
    return {"abs_err": abs_err, "rel_err": rel_err, "max_err": max_err, "time_s": elapsed}


# Lloyd-Max 16 级 (int4) 最优量化器针对标准正态 N(0,1) 的重构点 (单位: σ)。
# 由 quant_r_t.py 迭代求得,对称零均值。使用时按列 std 缩放。
_LLOYD_MAX_INT4_LEVELS_STD = [
    -2.7344, -2.0688, -1.6178, -1.2552, -0.9413, -0.6564, -0.3881, -0.1280,
    +0.1278, +0.3872, +0.6558, +0.9421, +1.2556, +1.6189, +2.0750, +2.7430,
]


def quantize_uniform_clip_sigma_per_col(x: torch.Tensor, levels: int = 16,
                                        clip_sigmas: float = 2.5) -> torch.Tensor:
    """按列均匀量化,动态范围裁剪到 [mean ± clip_sigmas·std] (非 min-max)。

    针对 N(0, σ²) 分布,均匀量化的最优裁剪范围约为 ±2.5σ (见 quant_r_t.py),
    比 min-max (常达 ±4σ 以上) 更少浪费码字。levels=16 即 int4。
    """
    mean = torch.zeros_like(x[:1])                   # 理论均值: 0
    std = torch.full_like(x[:1], 128 ** -0.5)        # 理论标准差: 1/sqrt(128)
    lo = mean - clip_sigmas * std
    hi = mean + clip_sigmas * std
    range_val = hi - lo
    scale = torch.where(range_val > 0, range_val / (levels - 1), torch.ones_like(range_val))
    q = torch.clamp(torch.round((x - lo) / scale), 0, levels - 1)
    return q * scale + lo


def quantize_lloyd_max_int4_per_col(x: torch.Tensor) -> torch.Tensor:
    """按列 Lloyd-Max 最优 int4 (16 级) 量化,针对高斯分布最小化 MSE。

    对每列先减均值、除以标准差归一化到 N(0,1),用预先求得的最优重构点
    (非均匀网格,中心密、尾部疏) 做最近邻量化,再还原尺度。
    """
    mean = torch.zeros_like(x[:1])                   # 理论均值: 0
    std = torch.full_like(x[:1], 128 ** -0.5)        # 理论标准差: 1/sqrt(128)
    std_safe = torch.where(std > 0, std, torch.ones_like(std))
    x_norm = (x - mean) / std_safe                   # 归一化到 N(0,1)

    y = torch.tensor(_LLOYD_MAX_INT4_LEVELS_STD, dtype=x.dtype, device=x.device)
    t = (y[1:] + y[:-1]) / 2                          # 内部判决边界 (中点)
    idx = torch.bucketize(x_norm, t)
    x_q = y[idx]                                      # 归一化尺度上的量化值
    return x_q * std_safe + mean                     # 还原尺度


def quantize_lloyd_max_per_col(x: torch.Tensor, bits: int) -> torch.Tensor:
    """按列 Lloyd-Max 最优 int`bits` 量化 (高斯非均匀网格),最小化 MSE。

    每列减均值、除标准差归一化到 N(0,1),用该 bit 数对应的 Lloyd-Max
    最优重构点做最近邻量化,再还原尺度。网格由 quant_r_t 求解并缓存。
    """
    mean = torch.zeros_like(x[:1])                   # 理论均值: 0
    std = torch.full_like(x[:1], 128 ** -0.5)        # 理论标准差: 1/sqrt(128)
    std_safe = torch.where(std > 0, std, torch.ones_like(std))
    x_norm = (x - mean) / std_safe

    y = _lloyd_max_grid_std(bits, dtype=x.dtype, device=x.device)
    t = (y[1:] + y[:-1]) / 2
    idx = torch.bucketize(x_norm, t)
    return y[idx] * std_safe + mean


def _quantize_companding_direct(
    x: torch.Tensor, min_val: torch.Tensor, max_val: torch.Tensor,
    bits: "int | torch.Tensor",
) -> torch.Tensor:
    """压缩到均匀域、取整为整数码,再通过逆变换反量化。

    bits 可为标量 (所有列同一 bit 数) 或形如 (k,) / (1, k) 的向量
    (每列独立 bit 数,mixed-precision)。向量时 num_levels 广播到各列。
    """
    sqrt2 = 2.0 ** 0.5
    eps = torch.finfo(x.dtype).eps

    if isinstance(bits, torch.Tensor):
        if torch.any(bits < 1):
            raise ValueError(f"bits 必须大于等于 1,得到 {bits.tolist()}")
        num_levels = (2.0 ** bits.to(x.dtype)).reshape(1, -1)   # (1, k) 逐列广播
    else:
        if bits < 1:
            raise ValueError(f"bits 必须大于等于 1,得到 {bits}")
        num_levels = float(2 ** bits)

    mu = torch.zeros_like(min_val)
    sigma = torch.full_like(min_val, x.shape[0] ** -0.5)
    companding_std = (3.0 ** 0.5) * sigma

    def normal_cdf(z: torch.Tensor) -> torch.Tensor:
        return 0.5 * (1.0 + torch.erf(z / sqrt2))

    cdf_min = normal_cdf((min_val - mu) / companding_std)
    cdf_max = normal_cdf((max_val - mu) / companding_std)
    cdf_span = (cdf_max - cdf_min).clamp_min(eps)

    compressed = (
        (normal_cdf((x - mu) / companding_std) - cdf_min) / cdf_span
    ).clamp(0.0, 1.0)
    # num_levels 可能是标量或 (1, k) 张量,clamp 的 max 两者均支持
    codes = torch.floor(compressed * num_levels).clamp(max=num_levels - 1)

    def inv(compressed_edge: torch.Tensor) -> torch.Tensor:
        cdf_q = (cdf_min + compressed_edge * cdf_span).clamp(eps, 1.0 - eps)
        return mu + companding_std * sqrt2 * torch.erfinv(2.0 * cdf_q - 1.0)

    # 每个 cell 的下/上边界映射回原始 x 空间,取其中点 (而非压缩域中点)
    x_lo = inv(codes / num_levels)
    x_hi = inv((codes + 1.0) / num_levels)
    return (x_lo + x_hi) / 2

def quantize_companding_per_col(
    x: torch.Tensor, bits: "int | torch.Tensor" = 4, clip_sigmas: float = 2.5
) -> torch.Tensor:
    """按列高斯压扩到均匀域量化,返回反量化后的浮点张量。

    非常数列先由截断点密度 CDF 映射到 [0,1],均匀取整为 int`bits`
    整数码,再通过逆 CDF 反变换,不构造边界表或重构值 LUT。
    常数列没有可量化的动态范围,直接保持原值。

    bits 可为标量 (所有列同一 bit 数),或形如 (k,) 的向量 (每列独立
    bit 数,mixed-precision);向量时长度须与 x 的列数一致。
    """
    n_cols = x.shape[1]
    if isinstance(bits, torch.Tensor):
        bits = bits.reshape(-1)
        if bits.numel() != n_cols:
            raise ValueError(f"bits 向量长度 {bits.numel()} 与列数 {n_cols} 不一致")

    min_val = x.amin(dim=0, keepdim=True)                     # (1, k)
    max_val = x.amax(dim=0, keepdim=True)                     # (1, k)
    variable_cols = (max_val > min_val).squeeze(0)            # (k,)
    if not torch.any(variable_cols):
        return x.clone()

    bits_arg = bits[variable_cols] if isinstance(bits, torch.Tensor) else bits
    xq = _quantize_companding_direct(
        x[:, variable_cols],
        min_val[:, variable_cols],
        max_val[:, variable_cols],
        bits_arg,
    )
    result = x.clone()
    result[:, variable_cols] = xq
    return result.contiguous()                                # (n, k)


def companding_reconstruction_error(
    A: torch.Tensor, U: torch.Tensor, s: torch.Tensor, Vt: torch.Tensor, r: int,
    bits: int = 4, clip_sigmas: float = 2.5,
) -> dict:
    """对 U、V 前 r 列做高斯最优压扩 int`bits` 量化后重构,打印误差。s 不量化。"""
    U_r = U[:, :r]
    V_r = Vt[:r, :].T
    s_r = s[:r]

    t0 = time.perf_counter()
    H = hadamard_matrix(U_r.shape[0], dtype=U_r.dtype, device=U_r.device)
    # H = torch.eye(U_r.shape[0], dtype=U_r.dtype, device=U_r.device)
    U_rot = H @ U_r
    V_rot = H @ V_r
    U_q = quantize_companding_per_col(U_rot, bits=bits, clip_sigmas=clip_sigmas)
    V_q = quantize_companding_per_col(V_rot, bits=bits, clip_sigmas=clip_sigmas)
    U_q = H @ U_q
    V_q = H @ V_q
    elapsed = time.perf_counter() - t0

    A_q = U_q @ torch.diag(s_r) @ V_q.T

    abs_err = torch.linalg.norm(A - A_q, ord="fro").item()
    rel_err = abs_err / torch.linalg.norm(A, ord="fro").item()
    max_err = (A - A_q).abs().max().item()

    print(f"\n=== 高斯最优压扩 int{bits} 按列量化重构误差 (clip ±{clip_sigmas}σ, U/V 量化, s 保留) ===")
    print(f"绝对误差 (Frobenius): {abs_err:.6e}")
    print(f"相对误差:            {rel_err:.6e}")
    print(f"最大元素误差:        {max_err:.6e}")
    print(f"量化用时:            {elapsed * 1e3:.3f} ms")
    print_quantization_bitrate(A, U, s, Vt, r, bits, bits, per_col="min_max")
    return {"abs_err": abs_err, "rel_err": rel_err, "max_err": max_err, "time_s": elapsed}


def uniform_intN_reconstruction_error(
    A: torch.Tensor, U: torch.Tensor, s: torch.Tensor, Vt: torch.Tensor, r: int, bits: int
) -> dict:
    """对 U、V 前 r 列做非对称 min-max 均匀 int`bits` 量化后重构,打印误差。s 不量化。"""
    U_r = U[:, :r]
    V_r = Vt[:r, :].T
    s_r = s[:r]

    t0 = time.perf_counter()
    bits_vec_U = torch.full((U_r.shape[1],), bits, dtype=torch.int64, device=U_r.device)
    bits_vec_V = torch.full((V_r.shape[1],), bits, dtype=torch.int64, device=V_r.device)
    U_q = quantize_per_col_bits(U_r, bits_vec_U)
    V_q = quantize_per_col_bits(V_r, bits_vec_V)
    elapsed = time.perf_counter() - t0

    A_q = U_q @ torch.diag(s_r) @ V_q.T

    abs_err = torch.linalg.norm(A - A_q, ord="fro").item()
    rel_err = abs_err / torch.linalg.norm(A, ord="fro").item()
    max_err = (A - A_q).abs().max().item()

    print(f"\n=== 均匀 min-max int{bits} 按列量化重构误差 (U, V 量化, s 保留) ===")
    print(f"绝对误差 (Frobenius): {abs_err:.6e}")
    print(f"相对误差:            {rel_err:.6e}")
    print(f"最大元素误差:        {max_err:.6e}")
    print(f"量化用时:            {elapsed * 1e3:.3f} ms")
    print_quantization_bitrate(A, U, s, Vt, r, bits, bits)
    return {"abs_err": abs_err, "rel_err": rel_err, "max_err": max_err, "time_s": elapsed}


def lloyd_max_intN_reconstruction_error(
    A: torch.Tensor, U: torch.Tensor, s: torch.Tensor, Vt: torch.Tensor, r: int, bits: int
) -> dict:
    """对 U、V 前 r 列做 Lloyd-Max 最优 int`bits` 量化后重构,打印误差。s 不量化。"""
    U_r = U[:, :r]
    V_r = Vt[:r, :].T
    s_r = s[:r]

    t0 = time.perf_counter()
    U_q = quantize_lloyd_max_per_col(U_r, bits)
    V_q = quantize_lloyd_max_per_col(V_r, bits)
    elapsed = time.perf_counter() - t0

    A_q = U_q @ torch.diag(s_r) @ V_q.T

    abs_err = torch.linalg.norm(A - A_q, ord="fro").item()
    rel_err = abs_err / torch.linalg.norm(A, ord="fro").item()
    max_err = (A - A_q).abs().max().item()

    print(f"\n=== Lloyd-Max 最优 int{bits} 按列量化重构误差 (U, V 量化, s 保留) ===")
    print(f"绝对误差 (Frobenius): {abs_err:.6e}")
    print(f"相对误差:            {rel_err:.6e}")
    print(f"最大元素误差:        {max_err:.6e}")
    print(f"量化用时:            {elapsed * 1e3:.3f} ms")
    print_quantization_bitrate(A, U, s, Vt, r, bits, bits, per_col=None)
    return {"abs_err": abs_err, "rel_err": rel_err, "max_err": max_err, "time_s": elapsed}


def quantize_kmeans_per_col(x: torch.Tensor, bits: int = 4, iters: int = 50,
                            tol: float = 1e-7) -> torch.Tensor:
    """按列 1-D k-means 聚类构建量化网格,返回反量化后的浮点张量 (向量化,全列并行)。

    x: (n, k) — 每列独立聚类。层级数 L = 2^bits,对每列用列内实际数据
    做 k-means 得到 L 个聚类中心作为码本 (数据自适应,不假设高斯),
    再把每个元素映射到最近中心 (簇内质心) 完成量化重建。
    常数列 (max==min) 聚类中心退化为同一值,重建即原值。

    向量化:码本 centers (k, L),用 batched searchsorted 做最近邻分配,
    scatter_add 按簇求和/计数并行更新质心,替代逐列 Python 循环。
    """
    n = x.shape[0]
    L = min(2 ** bits, n)
    xt = x.transpose(0, 1).contiguous()                      # (k, n)

    # 分位数 (等概率点) 初始化码本: (L, k) -> (k, L)
    qs = torch.linspace(0, 1, L, dtype=x.dtype, device=x.device)
    centers = torch.quantile(xt, qs, dim=1).transpose(0, 1).contiguous()  # (k, L)

    ones = torch.ones_like(xt)                               # (k, n) 计数用
    for _ in range(iters):
        bnd = (centers[:, 1:] + centers[:, :-1]) / 2         # (k, L-1) 判决边界
        idx = torch.searchsorted(bnd, xt)                    # (k, n) -> [0, L-1]
        sums = torch.zeros_like(centers).scatter_add_(1, idx, xt)     # 簇内和
        cnts = torch.zeros_like(centers).scatter_add_(1, idx, ones)   # 簇内计数
        new_centers = torch.where(cnts > 0, sums / cnts.clamp_min(1), centers)  # 空簇保留
        new_centers = torch.sort(new_centers, dim=1).values  # 保持升序
        shift = (new_centers - centers).abs().max()
        centers = new_centers
        if shift < tol:
            break

    # 最终分配 + 反量化
    bnd = (centers[:, 1:] + centers[:, :-1]) / 2
    idx = torch.searchsorted(bnd, xt)                        # (k, n)
    xq = torch.gather(centers, 1, idx)                       # (k, n)
    return xq.transpose(0, 1).contiguous()                   # (n, k)


def quantize_kmeans_per_tensor(x: torch.Tensor, bits: int = 4, iters: int = 50,
                               tol: float = 1e-7) -> torch.Tensor:
    """整张量 (per-tensor) 1-D k-means 聚类量化,返回反量化后的浮点张量。

    与 per-column 不同:所有元素共用一套 L = 2^bits 个聚类中心 (单一码本),
    只需存 L 个中心 + 每元素索引,元数据开销远小于逐列码本;代价是无法
    适配各列不同的动态范围。用分位数初始化,scatter_add 并行更新质心。
    """
    L = min(2 ** bits, x.numel())
    flat = x.reshape(-1)                                      # (n*k,)

    qs = torch.linspace(0, 1, L, dtype=x.dtype, device=x.device)
    centers = torch.quantile(flat, qs)                       # (L,) 分位数初始化

    ones = torch.ones_like(flat)
    for _ in range(iters):
        bnd = (centers[1:] + centers[:-1]) / 2               # (L-1,) 判决边界
        idx = torch.bucketize(flat, bnd)                     # (n*k,) -> [0, L-1]
        sums = torch.zeros_like(centers).scatter_add_(0, idx, flat)
        cnts = torch.zeros_like(centers).scatter_add_(0, idx, ones)
        new_centers = torch.where(cnts > 0, sums / cnts.clamp_min(1), centers)
        new_centers = torch.sort(new_centers).values         # 保持升序
        shift = (new_centers - centers).abs().max()
        centers = new_centers
        if shift < tol:
            break

    bnd = (centers[1:] + centers[:-1]) / 2
    idx = torch.bucketize(flat, bnd)
    return centers[idx].reshape_as(x)


def kmeans_per_tensor_reconstruction_error(
    A: torch.Tensor, U: torch.Tensor, s: torch.Tensor, Vt: torch.Tensor, r: int, bits: int = 4
) -> dict:
    """对 U、V 前 r 列做 per-tensor k-means (单一共享码本) int`bits` 量化后重构,打印误差。s 不量化。"""
    U_r = U[:, :r]            # (n, r)
    V_r = Vt[:r, :].T         # (n, r)
    s_r = s[:r]

    t0 = time.perf_counter()
    U_q = quantize_kmeans_per_tensor(U_r, bits=bits)
    V_q = quantize_kmeans_per_tensor(V_r, bits=bits)
    elapsed = time.perf_counter() - t0

    A_q = U_q @ torch.diag(s_r) @ V_q.T

    abs_err = torch.linalg.norm(A - A_q, ord="fro").item()
    rel_err = abs_err / torch.linalg.norm(A, ord="fro").item()
    max_err = (A - A_q).abs().max().item()

    print(f"\n=== per-tensor k-means 共享码本 int{bits} 量化重构误差 (U, V 量化, s 保留) ===")
    print(f"绝对误差 (Frobenius): {abs_err:.6e}")
    print(f"相对误差:            {rel_err:.6e}")
    print(f"最大元素误差:        {max_err:.6e}")
    print(f"量化用时:            {elapsed * 1e3:.3f} ms")
    print_quantization_bitrate(A, U, s, Vt, r, bits, bits, kmeans="per_tensor")
    return {"abs_err": abs_err, "rel_err": rel_err, "max_err": max_err, "time_s": elapsed}


def kmeans_intN_reconstruction_error(
    A: torch.Tensor, U: torch.Tensor, s: torch.Tensor, Vt: torch.Tensor, r: int, bits: int = 4
) -> dict:
    """对 U、V 前 r 列做 per-column k-means 聚类网格 int`bits` 量化后重构,打印误差。s 不量化。"""
    U_r = U[:, :r]            # (n, r)
    V_r = Vt[:r, :].T         # (n, r)
    s_r = s[:r]

    t0 = time.perf_counter()
    H = hadamard_matrix(U_r.shape[0], dtype=U_r.dtype, device=U_r.device)
    # H = torch.eye(U_r.shape[0], dtype=U_r.dtype, device=U_r.device)
    U_rot = H @ U_r
    V_rot = H @ V_r
    U_q = quantize_kmeans_per_col(U_rot, bits=bits)
    V_q = quantize_kmeans_per_col(V_rot, bits=bits)
    U_q = H @ U_q
    V_q = H @ V_q
    elapsed = time.perf_counter() - t0

    A_q = U_q @ torch.diag(s_r) @ V_q.T

    abs_err = torch.linalg.norm(A - A_q, ord="fro").item()
    rel_err = abs_err / torch.linalg.norm(A, ord="fro").item()
    max_err = (A - A_q).abs().max().item()

    print(f"\n=== per-column k-means 聚类网格 int{bits} 量化重构误差 (U, V 量化, s 保留) ===")
    print(f"绝对误差 (Frobenius): {abs_err:.6e}")
    print(f"相对误差:            {rel_err:.6e}")
    print(f"最大元素误差:        {max_err:.6e}")
    print(f"量化用时:            {elapsed * 1e3:.3f} ms")
    print_quantization_bitrate(A, U, s, Vt, r, bits, bits, kmeans="per_col")
    return {"abs_err": abs_err, "rel_err": rel_err, "max_err": max_err, "time_s": elapsed}


def quantize_int4_per_col(x: torch.Tensor) -> torch.Tensor:
    """非对称 int4 按列量化,返回反量化后的浮点张量。

    x: (n, k) — 每列(长度 n)独立计算 scale 与 zero_point。
    非对称 int4: 4 bit 无符号,取值范围 [0, 15],共 16 个层级。
      scale = (max - min) / 15
      zp    = clamp(round(-min / scale), 0, 15)   # 使 x=0 映射到 zp 附近
      q     = clamp(round(x / scale + zp), 0, 15)
      deq   = (q - zp) * scale
    相较对称量化,非对称利用完整 [min, max] 区间,对非零均值分布更高效。
    """
    return quantize_per_col_bits(x, bits=4)


def quantize_int6_per_col(x: torch.Tensor) -> torch.Tensor:
    """非对称 int6 按列量化,返回反量化后的浮点张量。

    x: (n, k) — 每列(长度 n)独立计算 scale 与 zero_point。
    非对称 int6: 6 bit 无符号,取值范围 [0, 63],共 64 个层级。
      scale = (max - min) / 63
      zp    = clamp(round(-min / scale), 0, 63)
      q     = clamp(round(x / scale + zp), 0, 63)
      deq   = (q - zp) * scale
    与 int4 同为均匀网格,层级数从 16 增至 64,量化步长约缩小 4x。
    """
    return quantize_per_col_bits(x, bits=6)


def quantize_int5_per_col(x: torch.Tensor) -> torch.Tensor:
    """非对称 int5 按列量化,返回反量化后的浮点张量。

    5 bit 无符号,取值范围 [0, 31],共 32 个层级。scale = (max - min) / 31。
    """
    return quantize_per_col_bits(x, bits=5)


def quantize_int7_per_col(x: torch.Tensor) -> torch.Tensor:
    """非对称 int7 按列量化。7 bit 无符号,[0, 127],128 个层级,scale = range / 127。"""
    return quantize_per_col_bits(x, bits=7)


def quantize_int8_per_col(x: torch.Tensor) -> torch.Tensor:
    """非对称 int8 按列量化。8 bit 无符号,[0, 255],256 个层级,scale = range / 255。"""
    return quantize_per_col_bits(x, bits=8)


def int7_quantized_reconstruction_error(
    A: torch.Tensor, U: torch.Tensor, s: torch.Tensor, Vt: torch.Tensor, r: int
) -> dict:
    """对 U、V 前 r 个奇异向量按列做非对称 int7 量化后重构,打印误差。s 不量化。"""
    U_r = U[:, :r]
    V_r = Vt[:r, :].T
    s_r = s[:r]
    t0 = time.perf_counter()
    U_q = quantize_int7_per_col(U_r)
    V_q = quantize_int7_per_col(V_r)
    elapsed = time.perf_counter() - t0
    A_q = U_q @ torch.diag(s_r) @ V_q.T
    abs_err = torch.linalg.norm(A - A_q, ord="fro").item()
    rel_err = abs_err / torch.linalg.norm(A, ord="fro").item()
    max_err = (A - A_q).abs().max().item()
    print("\n=== 非对称 int7 按列量化重构误差 (U, V 量化, s 保留) ===")
    print(f"绝对误差 (Frobenius): {abs_err:.6e}")
    print(f"相对误差:            {rel_err:.6e}")
    print(f"最大元素误差:        {max_err:.6e}")
    print(f"量化用时:            {elapsed * 1e3:.3f} ms")
    print_quantization_bitrate(A, U, s, Vt, r, 7, 7)
    return {"abs_err": abs_err, "rel_err": rel_err, "max_err": max_err, "time_s": elapsed}


def int8_quantized_reconstruction_error(
    A: torch.Tensor, U: torch.Tensor, s: torch.Tensor, Vt: torch.Tensor, r: int
) -> dict:
    """对 U、V 前 r 个奇异向量按列做非对称 int8 量化后重构,打印误差。s 不量化。"""
    U_r = U[:, :r]
    V_r = Vt[:r, :].T
    s_r = s[:r]
    t0 = time.perf_counter()
    U_q = quantize_int8_per_col(U_r)
    V_q = quantize_int8_per_col(V_r)
    elapsed = time.perf_counter() - t0
    A_q = U_q @ torch.diag(s_r) @ V_q.T
    abs_err = torch.linalg.norm(A - A_q, ord="fro").item()
    rel_err = abs_err / torch.linalg.norm(A, ord="fro").item()
    max_err = (A - A_q).abs().max().item()
    print("\n=== 非对称 int8 按列量化重构误差 (U, V 量化, s 保留) ===")
    print(f"绝对误差 (Frobenius): {abs_err:.6e}")
    print(f"相对误差:            {rel_err:.6e}")
    print(f"最大元素误差:        {max_err:.6e}")
    print(f"量化用时:            {elapsed * 1e3:.3f} ms")
    print_quantization_bitrate(A, U, s, Vt, r, 8, 8)
    return {"abs_err": abs_err, "rel_err": rel_err, "max_err": max_err, "time_s": elapsed}


def quantize_per_col_bits(x: torch.Tensor, bits: int | torch.Tensor) -> torch.Tensor:
    """非对称按列量化,每列使用独立的 bit 数(mixed-precision)。

    bits 为 1~8 的整数或每列整数向量。scale 保存为 float32,zero 保存为 uint8。
    用保存后的参数量化和反量化,返回 x.dtype;低精度输入在 float32 中运算。
    无法用正有限 float32 表示的 scale 明确报错,不静默溢出或下溢。
    """
    if x.ndim != 2 or not x.is_floating_point() or x.numel() == 0:
        raise ValueError("expected a nonempty floating-point matrix")
    if not torch.isfinite(x).all():
        raise ValueError("expected finite matrix values")
    if isinstance(bits, torch.Tensor):
        if bits.numel() != x.shape[1] or bits.is_floating_point() or bits.is_complex():
            raise ValueError("bits must be an integer vector with one entry per column")
        if torch.any((bits < 1) | (bits > 8)):
            raise ValueError("uint8 zero points require bit widths from 1 to 8")
    elif not isinstance(bits, int) or not 1 <= bits <= 8:
        raise ValueError("bits must be an integer from 1 to 8")
    values = x if x.dtype == torch.float64 else x.float()
    min_val = values.amin(dim=0, keepdim=True)
    max_val = values.amax(dim=0, keepdim=True)
    range_val = max_val - min_val
    bit_values = torch.as_tensor(bits, dtype=values.dtype, device=x.device).reshape(1, -1)
    levels = 2.0 ** bit_values - 1.0
    scale = torch.where(range_val > 0, range_val / levels, torch.ones_like(range_val)).float()
    if not torch.isfinite(scale).all() or torch.any(scale <= 0):
        raise ValueError("scale is outside the positive finite float32 range")
    restored_scale = scale.to(values.dtype)
    zero = torch.clamp(torch.round(-min_val / restored_scale), torch.zeros_like(levels), levels).to(torch.uint8)
    restored_zero = zero.to(values.dtype)
    codes = torch.clamp(torch.round(values / restored_scale + restored_zero), torch.zeros_like(levels), levels)
    return ((codes - restored_zero) * restored_scale).to(x.dtype)


def allocate_bits_by_singular_value(
    s_r: torch.Tensor, avg_bits: float, bit_min: int = 3, bit_max: int = 8
) -> torch.Tensor:
    """按奇异值大小分配每列 bit 数,使平均 bit 接近 avg_bits。

    依据最优 bit 分配: b_i ∝ log2(s_i) + 常数。
    以 avg_bits 为中心、log2(s_i) 相对均值的偏移为增量,四舍五入并裁剪到 [bit_min, bit_max]。
    """
    log_s = torch.log2(s_r.clamp_min(torch.finfo(s_r.dtype).tiny))
    bits_f = avg_bits + (log_s - log_s.mean())
    bits = torch.clamp(torch.round(bits_f), bit_min, bit_max).to(torch.int64)
    return bits


def mixed_precision_by_singular_value_reconstruction_error(
    A: torch.Tensor, U: torch.Tensor, s: torch.Tensor, Vt: torch.Tensor, r: int,
    avg_bits: float = 4.0,
) -> dict:
    """按奇异值大小对各列分配量化精度 (mixed-precision) 后重构,打印误差。

    大奇异值列(对重构贡献大)分配更多 bit,小奇异值列分配更少 bit,
    平均 bit 约束到 avg_bits,便于与均匀 int(avg_bits) 公平对比。
    U、V 使用同一套按 s 得到的 bit 分配。
    """
    U_r = U[:, :r]            # (n, r)
    V_r = Vt[:r, :].T         # (n, r)
    s_r = s[:r]

    t0 = time.perf_counter()
    bits = allocate_bits_by_singular_value(s_r, avg_bits)
    H = hadamard_matrix(U_r.shape[0], dtype=U_r.dtype, device=U_r.device)
    # H = torch.eye(U_r.shape[0], dtype=U_r.dtype, device=U_r.device)
    U_rot = H @ U_r
    V_rot = H @ V_r
    # U_q = quantize_per_col_bits(U_rot, bits)
    # V_q = quantize_per_col_bits(V_rot, bits)
    U_q = quantize_companding_per_col(U_rot, bits=bits)
    V_q = quantize_companding_per_col(V_rot, bits=bits)
    U_q = H @ U_q
    V_q = H @ V_q
    elapsed = time.perf_counter() - t0

    A_q = U_q @ torch.diag(s_r) @ V_q.T

    abs_err = torch.linalg.norm(A - A_q, ord="fro").item()
    rel_err = abs_err / torch.linalg.norm(A, ord="fro").item()
    max_err = (A - A_q).abs().max().item()

    print(f"\n=== 按奇异值 mixed-precision 量化重构误差 (平均 {avg_bits} bit, s 保留) ===")
    print(f"每列 bit 分配: {bits.tolist()}  (实际平均 {bits.to(torch.float64).mean():.2f})")
    print(f"绝对误差 (Frobenius): {abs_err:.6e}")
    print(f"相对误差:            {rel_err:.6e}")
    print(f"最大元素误差:        {max_err:.6e}")
    print(f"量化用时:            {elapsed * 1e3:.3f} ms")
    print_quantization_bitrate(A, U, s, Vt, r, bits, bits, per_col="min_max")
    return {"abs_err": abs_err, "rel_err": rel_err, "max_err": max_err, "time_s": elapsed}

def mixed_precision_companding_update_V_reconstruction(
    A: torch.Tensor, U: torch.Tensor, s: torch.Tensor, Vt: torch.Tensor, r: int,
    avg_bits: float = 4.0,
) -> dict:
    """按奇异值分配 bits,量化 U 后最小二乘更新 V,再量化 V。

    U/V 均使用 Hadamard 旋转和压扩量化,共享原始 s 决定的 bit 分配。
    固定 U_q 和 s,求解 min ||A - (U_q * s) @ V_new.T||_F。
    计时包含 bit 分配、正反旋转、量化及最小二乘,不含最终重构和误差计算。
    """
    U_r = U[:, :r]
    V_r = Vt[:r, :].T
    s_r = s[:r]

    t0 = time.perf_counter()
    bits = allocate_bits_by_singular_value(s_r, avg_bits)
    hadamard_U = hadamard_matrix(U_r.shape[0], dtype=U_r.dtype, device=U_r.device)
    hadamard_V = hadamard_matrix(V_r.shape[0], dtype=V_r.dtype, device=V_r.device)
    U_rot = hadamard_U @ U_r
    U_q = hadamard_U.T @ quantize_companding_per_col(U_rot, bits=bits)

    V_new = torch.linalg.lstsq(U_q * s_r, A).solution.T
    V_rot = hadamard_V @ V_new
    V_q = hadamard_V.T @ quantize_companding_per_col(V_rot, bits=bits)
    elapsed = time.perf_counter() - t0

    A_q = (U_q * s_r) @ V_q.T
    # abs_err = torch.linalg.norm(A - A_q, ord="fro").item()
    # rel_err = abs_err / torch.linalg.norm(A, ord="fro").item()
    # max_err = (A - A_q).abs().max().item()

    # print(f"\n=== mixed-precision U 量化 + V 最小二乘更新 + V 再量化 (平均 {avg_bits} bit, s 保留) ===")
    # print(f"每列 bit 分配: {bits.tolist()}  (实际平均 {bits.to(torch.float64).mean():.2f})")
    # print(f"绝对误差 (Frobenius): {abs_err:.6e}")
    # print(f"相对误差:            {rel_err:.6e}")
    # print(f"最大元素误差:        {max_err:.6e}")
    # print(f"量化用时:            {elapsed * 1e3:.3f} ms")
    # print_quantization_bitrate(A, U, s, Vt, r, bits, bits, per_col="min_max")
    return A_q

def mixed_precision_U_quant_update_V_then_quant_V_reconstruction_error(
    A: torch.Tensor, U: torch.Tensor, s: torch.Tensor, Vt: torch.Tensor, r: int,
    avg_bits: float = 4.0,
) -> dict:
    """按奇异值分配 bits,量化 U 后最小二乘更新 V,再量化 V。

    U/V 均使用 Hadamard 旋转和压扩量化,共享原始 s 决定的 bit 分配。
    固定 U_q 和 s,求解 min ||A - (U_q * s) @ V_new.T||_F。
    计时包含 bit 分配、正反旋转、量化及最小二乘,不含最终重构和误差计算。
    """
    U_r = U[:, :r]
    V_r = Vt[:r, :].T
    s_r = s[:r]

    t0 = time.perf_counter()
    bits = allocate_bits_by_singular_value(s_r, avg_bits)
    hadamard_U = hadamard_matrix(U_r.shape[0], dtype=U_r.dtype, device=U_r.device)
    hadamard_V = hadamard_matrix(V_r.shape[0], dtype=V_r.dtype, device=V_r.device)
    U_rot = hadamard_U @ U_r
    U_q = hadamard_U.T @ quantize_companding_per_col(U_rot, bits=bits)

    V_new = torch.linalg.lstsq(U_q * s_r, A).solution.T
    V_rot = hadamard_V @ V_new
    V_q = hadamard_V.T @ quantize_companding_per_col(V_rot, bits=bits)
    elapsed = time.perf_counter() - t0

    A_q = (U_q * s_r) @ V_q.T
    abs_err = torch.linalg.norm(A - A_q, ord="fro").item()
    rel_err = abs_err / torch.linalg.norm(A, ord="fro").item()
    max_err = (A - A_q).abs().max().item()

    print(f"\n=== mixed-precision U 量化 + V 最小二乘更新 + V 再量化 (平均 {avg_bits} bit, s 保留) ===")
    print(f"每列 bit 分配: {bits.tolist()}  (实际平均 {bits.to(torch.float64).mean():.2f})")
    print(f"绝对误差 (Frobenius): {abs_err:.6e}")
    print(f"相对误差:            {rel_err:.6e}")
    print(f"最大元素误差:        {max_err:.6e}")
    print(f"量化用时:            {elapsed * 1e3:.3f} ms")
    print_quantization_bitrate(A, U, s, Vt, r, bits, bits, per_col="min_max")
    return {"abs_err": abs_err, "rel_err": rel_err, "max_err": max_err, "time_s": elapsed}


def mixed_precision_alternating_quant_UVU_reconstruction_error(
    A: torch.Tensor, U: torch.Tensor, s: torch.Tensor, Vt: torch.Tensor, r: int,
    avg_bits: float = 4.0,
) -> dict:
    """按奇异值分配 bits,交替执行 U 量化、V 更新并量化、U 更新并再量化。

    固定 s,两次最小二乘均以原始 A 为目标,三次量化共享 bit 分配。
    U/V 均使用 Hadamard 旋转和压扩量化。
    计时包含 bit 分配、正反旋转、量化及最小二乘,不含最终重构和误差计算。
    """
    U_r = U[:, :r]
    V_r = Vt[:r, :].T
    s_r = s[:r]

    t0 = time.perf_counter()
    bits = allocate_bits_by_singular_value(s_r, avg_bits)
    hadamard_U = hadamard_matrix(U_r.shape[0], dtype=U_r.dtype, device=U_r.device)
    hadamard_V = hadamard_matrix(V_r.shape[0], dtype=V_r.dtype, device=V_r.device)
    U_rot = hadamard_U @ U_r
    U_q = hadamard_U.T @ quantize_companding_per_col(U_rot, bits=bits)

    V_new = torch.linalg.lstsq(U_q * s_r, A).solution.T
    V_rot = hadamard_V @ V_new
    V_q = hadamard_V.T @ quantize_companding_per_col(V_rot, bits=bits)

    U_new = torch.linalg.lstsq(V_q * s_r, A.T).solution.T
    U_rot = hadamard_U @ U_new
    U_q = hadamard_U.T @ quantize_companding_per_col(U_rot, bits=bits)
    elapsed = time.perf_counter() - t0

    A_q = (U_q * s_r) @ V_q.T
    abs_err = torch.linalg.norm(A - A_q, ord="fro").item()
    rel_err = abs_err / torch.linalg.norm(A, ord="fro").item()
    max_err = (A - A_q).abs().max().item()

    print(f"\n=== mixed-precision 交替量化 U->更新V->量化V->更新U->量化U (平均 {avg_bits} bit, s 保留) ===")
    print(f"每列 bit 分配: {bits.tolist()}  (实际平均 {bits.to(torch.float64).mean():.2f})")
    print(f"绝对误差 (Frobenius): {abs_err:.6e}")
    print(f"相对误差:            {rel_err:.6e}")
    print(f"最大元素误差:        {max_err:.6e}")
    print(f"量化用时:            {elapsed * 1e3:.3f} ms")
    print_quantization_bitrate(A, U, s, Vt, r, bits, bits, per_col="min_max")
    return {"abs_err": abs_err, "rel_err": rel_err, "max_err": max_err, "time_s": elapsed}


def int5_quantized_reconstruction_error(
    A: torch.Tensor, U: torch.Tensor, s: torch.Tensor, Vt: torch.Tensor, r: int
) -> dict:
    """对 U、V 的前 r 个奇异向量按列做非对称 int5 量化后重构,打印误差。奇异值 s 不量化。"""
    U_r = U[:, :r]            # (n, r) 左奇异向量
    V_r = Vt[:r, :].T         # (n, r) 右奇异向量
    s_r = s[:r]

    t0 = time.perf_counter()
    U_q = quantize_int5_per_col(U_r)
    V_q = quantize_int5_per_col(V_r)
    elapsed = time.perf_counter() - t0

    A_q = U_q @ torch.diag(s_r) @ V_q.T

    abs_err = torch.linalg.norm(A - A_q, ord="fro").item()
    rel_err = abs_err / torch.linalg.norm(A, ord="fro").item()
    max_err = (A - A_q).abs().max().item()

    print("\n=== 非对称 int5 按列量化重构误差 (U, V 量化, s 保留) ===")
    print(f"绝对误差 (Frobenius): {abs_err:.6e}")
    print(f"相对误差:            {rel_err:.6e}")
    print(f"最大元素误差:        {max_err:.6e}")
    print(f"量化用时:            {elapsed * 1e3:.3f} ms")
    print_quantization_bitrate(A, U, s, Vt, r, 5, 5)
    return {"abs_err": abs_err, "rel_err": rel_err, "max_err": max_err, "time_s": elapsed}


def int6_quantized_reconstruction_error(
    A: torch.Tensor, U: torch.Tensor, s: torch.Tensor, Vt: torch.Tensor, r: int
) -> dict:
    """对 U、V 的前 r 个奇异向量按列做非对称 int6 量化后重构,打印误差。奇异值 s 不量化。"""
    U_r = U[:, :r]            # (n, r) 左奇异向量
    V_r = Vt[:r, :].T         # (n, r) 右奇异向量
    s_r = s[:r]

    t0 = time.perf_counter()
    U_q = quantize_int6_per_col(U_r)
    V_q = quantize_int6_per_col(V_r)
    elapsed = time.perf_counter() - t0

    A_q = U_q @ torch.diag(s_r) @ V_q.T

    abs_err = torch.linalg.norm(A - A_q, ord="fro").item()
    rel_err = abs_err / torch.linalg.norm(A, ord="fro").item()
    max_err = (A - A_q).abs().max().item()

    print("\n=== 非对称 int6 按列量化重构误差 (U, V 量化, s 保留) ===")
    print(f"绝对误差 (Frobenius): {abs_err:.6e}")
    print(f"相对误差:            {rel_err:.6e}")
    print(f"最大元素误差:        {max_err:.6e}")
    print(f"量化用时:            {elapsed * 1e3:.3f} ms")
    print_quantization_bitrate(A, U, s, Vt, r, 6, 6)
    return {"abs_err": abs_err, "rel_err": rel_err, "max_err": max_err, "time_s": elapsed}


def uniform_clip_sigma_int4_reconstruction_error(
    A: torch.Tensor, U: torch.Tensor, s: torch.Tensor, Vt: torch.Tensor, r: int,
    clip_sigmas: float = 2.5,
) -> dict:
    """对 U、V 前 r 列做均匀裁剪 ±clip_sigmas·σ 的 int4 量化后重构,打印误差。s 不量化。"""
    U_r = U[:, :r]            # (n, r)
    V_r = Vt[:r, :].T         # (n, r)
    s_r = s[:r]

    t0 = time.perf_counter()
    U_q = quantize_uniform_clip_sigma_per_col(U_r, levels=16, clip_sigmas=clip_sigmas)
    V_q = quantize_uniform_clip_sigma_per_col(V_r, levels=16, clip_sigmas=clip_sigmas)
    elapsed = time.perf_counter() - t0

    A_q = U_q @ torch.diag(s_r) @ V_q.T

    abs_err = torch.linalg.norm(A - A_q, ord="fro").item()
    rel_err = abs_err / torch.linalg.norm(A, ord="fro").item()
    max_err = (A - A_q).abs().max().item()

    print(f"\n=== 均匀裁剪 ±{clip_sigmas}σ int4 按列量化重构误差 (U, V 量化, s 保留) ===")
    print(f"绝对误差 (Frobenius): {abs_err:.6e}")
    print(f"相对误差:            {rel_err:.6e}")
    print(f"最大元素误差:        {max_err:.6e}")
    print(f"量化用时:            {elapsed * 1e3:.3f} ms")
    print_quantization_bitrate(A, U, s, Vt, r, 4, 4, per_col=None)
    return {"abs_err": abs_err, "rel_err": rel_err, "max_err": max_err, "time_s": elapsed}


def lloyd_max_int4_reconstruction_error(
    A: torch.Tensor, U: torch.Tensor, s: torch.Tensor, Vt: torch.Tensor, r: int
) -> dict:
    """对 U、V 前 r 列做 Lloyd-Max 最优 int4 (高斯非均匀网格) 量化后重构,打印误差。s 不量化。"""
    U_r = U[:, :r]            # (n, r)
    V_r = Vt[:r, :].T         # (n, r)
    s_r = s[:r]
    t0 = time.perf_counter()
    H = hadamard_matrix(U_r.shape[0], dtype=U_r.dtype, device=U_r.device)
    # H = torch.eye(U_r.shape[0], dtype=U_r.dtype, device=U_r.device)
    U_rot = H @ U_r
    V_rot = H @ V_r
    U_q = quantize_lloyd_max_int4_per_col(U_rot)
    V_q = quantize_lloyd_max_int4_per_col(V_rot)
    U_q = H @ U_q
    V_q = H @ V_q
    elapsed = time.perf_counter() - t0

    A_q = U_q @ torch.diag(s_r) @ V_q.T

    abs_err = torch.linalg.norm(A - A_q, ord="fro").item()
    rel_err = abs_err / torch.linalg.norm(A, ord="fro").item()
    max_err = (A - A_q).abs().max().item()

    print("\n=== Lloyd-Max 最优 int4 按列量化重构误差 (U, V 量化, s 保留) ===")
    print(f"绝对误差 (Frobenius): {abs_err:.6e}")
    print(f"相对误差:            {rel_err:.6e}")
    print(f"最大元素误差:        {max_err:.6e}")
    print(f"量化用时:            {elapsed * 1e3:.3f} ms")
    print_quantization_bitrate(A, U, s, Vt, r, 4, 4, per_col=None)
    return {"abs_err": abs_err, "rel_err": rel_err, "max_err": max_err, "time_s": elapsed}


def int4_quantized_reconstruction_error(
    A: torch.Tensor, U: torch.Tensor, s: torch.Tensor, Vt: torch.Tensor, r: int
) -> dict:
    """对 U、V 的前 r 个奇异向量按列做 int4 量化后重构,打印误差。

    量化对象:
      - U[:, :r] (n, r): 左奇异向量,按列(长度 n)量化
      - Vt[:r, :].T (n, r): 右奇异向量,按列(长度 n)量化
    奇异值 s 不量化。
    """
    U_r = U[:, :r]            # (n, r) 左奇异向量
    V_r = Vt[:r, :].T         # (n, r) 右奇异向量
    s_r = s[:r]

    t0 = time.perf_counter()
    U_q = quantize_int4_per_col(U_r)
    V_q = quantize_int4_per_col(V_r)
    elapsed = time.perf_counter() - t0

    A_q = U_q @ torch.diag(s_r) @ V_q.T

    abs_err = torch.linalg.norm(A - A_q, ord="fro").item()
    rel_err = abs_err / torch.linalg.norm(A, ord="fro").item()
    max_err = (A - A_q).abs().max().item()

    print("\n=== 非对称 int4 按列量化重构误差 (U, V 量化, s 保留) ===")
    print(f"绝对误差 (Frobenius): {abs_err:.6e}")
    print(f"相对误差:            {rel_err:.6e}")
    print(f"最大元素误差:        {max_err:.6e}")
    print(f"量化用时:            {elapsed * 1e3:.3f} ms")
    print_quantization_bitrate(A, U, s, Vt, r, 4, 4)
    return {"abs_err": abs_err, "rel_err": rel_err, "max_err": max_err, "time_s": elapsed}


# def gptq_int4_reconstruction_error(
#     A: torch.Tensor, U: torch.Tensor, s: torch.Tensor, Vt: torch.Tensor, r: int
# ) -> dict:
#     """用重构乘法的输入校准 GPTQ int4 量化 U、V,保留 s。"""
#     U_r = U[:, :r]
#     V_r = Vt[:r, :].T
#     s_r = s[:r]

#     t0 = time.perf_counter()
#     U_q = quantize_weight(U_r, V_r * s_r, bits=4)
#     V_q = quantize_weight(V_r, U_q * s_r, bits=4)
#     elapsed = time.perf_counter() - t0

#     A_q = (U_q * s_r) @ V_q.T
#     abs_err = torch.linalg.norm(A - A_q, ord="fro").item()
#     rel_err = abs_err / torch.linalg.norm(A, ord="fro").item()
#     max_err = (A - A_q).abs().max().item()

#     print("\n=== GPTQ 非对称 int4 重构误差 (U, V 量化, s 保留) ===")
#     print(f"绝对误差 (Frobenius): {abs_err:.6e}")
#     print(f"相对误差:            {rel_err:.6e}")
#     print(f"最大元素误差:        {max_err:.6e}")
#     print(f"量化用时:            {elapsed * 1e3:.3f} ms")
#     print_quantization_bitrate(A, U, s, Vt, r, 4, 4, gptq=True)
#     return {"abs_err": abs_err, "rel_err": rel_err, "max_err": max_err, "time_s": elapsed}


def hadamard_matrix(n: int, dtype: torch.dtype, device: str = "cpu") -> torch.Tensor:
    """构造归一化 Walsh-Hadamard 矩阵 (n x n),n 必须为 2 的幂。

    递归构造 H_{2k} = [[H_k, H_k], [H_k, -H_k]],再除以 sqrt(n) 归一化。
    性质:对称且正交,H = H^T = H^{-1},故 H @ H = I。
    """
    assert n > 0 and (n & (n - 1)) == 0, f"n 必须是 2 的幂,得到 {n}"
    H = torch.ones((1, 1), dtype=dtype, device=device)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], dim=1), torch.cat([H, -H], dim=1)], dim=0)
    return H / (n ** 0.5)


def hadamard_int4_quantized_reconstruction_error(
    A: torch.Tensor, U: torch.Tensor, s: torch.Tensor, Vt: torch.Tensor, r: int
) -> dict:
    """Hadamard 旋转 + 非对称 int4 按列量化后重构,打印误差。

    流程 (左旋转,作用于 n=128 维):
      1. H = 归一化 128 阶 Walsh-Hadamard 矩阵 (正交对称, H^{-1}=H)
      2. U_rot = H @ U[:, :r]   # 旋转后能量在 128 维内重新分布,使列内峰值降低
         V_rot = H @ Vt[:r, :].T
      3. 对 U_rot、V_rot 各列做非对称 int4 量化,反量化得 U_q、V_q
      4. 反旋转:U_final = H @ U_q, V_final = H @ V_q  (因 H^{-1}=H)
      5. A_q = U_final @ diag(s[:r]) @ V_final.T

    旋转使每列分布更平坦,降低 max - min,从而减小量化 scale,提升精度。
    """
    n = A.shape[0]
    dtype = A.dtype
    device = A.device.type if hasattr(A.device, "type") else str(A.device)

    t0 = time.perf_counter()
    H = hadamard_matrix(n, dtype=dtype, device=device)

    U_r = U[:, :r]            # (n, r)
    V_r = Vt[:r, :].T         # (n, r)
    s_r = s[:r]

    # 旋转
    U_rot = H @ U_r
    V_rot = H @ V_r

    # 量化 + 反量化
    U_q = quantize_int4_per_col(U_rot)
    V_q = quantize_int4_per_col(V_rot)

    # 反旋转 (H^{-1} = H)
    U_final = H @ U_q
    V_final = H @ V_q
    elapsed = time.perf_counter() - t0

    A_q = U_final @ torch.diag(s_r) @ V_final.T

    abs_err = torch.linalg.norm(A - A_q, ord="fro").item()
    rel_err = abs_err / torch.linalg.norm(A, ord="fro").item()
    max_err = (A - A_q).abs().max().item()

    print("\n=== Hadamard 旋转 + 非对称 int4 按列量化重构误差 ===")
    print(f"绝对误差 (Frobenius): {abs_err:.6e}")
    print(f"相对误差:            {rel_err:.6e}")
    print(f"最大元素误差:        {max_err:.6e}")
    print(f"量化用时:            {elapsed * 1e3:.3f} ms")
    print_quantization_bitrate(A, U, s, Vt, r, 4, 4)
    return {"abs_err": abs_err, "rel_err": rel_err, "max_err": max_err, "time_s": elapsed}


def fp8_reconstruction_error(
    A: torch.Tensor, U: torch.Tensor, s: torch.Tensor, Vt: torch.Tensor, r: int
) -> dict:
    """U、V 转为 fp8 (e4m3) 存储后转回高精度重构,打印误差。不做整数量化。

    fp8 e4m3: 1 符号位 + 4 指数位 + 3 尾数位,共 8 bit 浮点。
      - 动态范围约 ±448,相对精度 ~2^-3 (尾数 3 位)
      - 奇异向量元素在 [-1, 1] 内,完全落在 e4m3 范围内
    与 int4 (4 bit, 16 个均匀层级) 对比:fp8 有 8 bit,且是浮点分布
    (小数值精度更高),预期误差显著更小。
    """
    U_r = U[:, :r]            # (n, r)
    V_r = Vt[:r, :].T         # (n, r)
    s_r = s[:r]

    t0 = time.perf_counter()
    fp8_dtype = torch.float8_e4m3fn
    # 转 fp8 存储,再转回高精度参与运算
    U_f8 = U_r.to(fp8_dtype).to(A.dtype)
    V_f8 = V_r.to(fp8_dtype).to(A.dtype)
    elapsed = time.perf_counter() - t0

    A_f8 = U_f8 @ torch.diag(s_r) @ V_f8.T

    abs_err = torch.linalg.norm(A - A_f8, ord="fro").item()
    rel_err = abs_err / torch.linalg.norm(A, ord="fro").item()
    max_err = (A - A_f8).abs().max().item()

    print("\n=== fp8 (e4m3) 精度转换重构误差 (U, V 转 fp8, s 保留) ===")
    print(f"绝对误差 (Frobenius): {abs_err:.6e}")
    print(f"相对误差:            {rel_err:.6e}")
    print(f"最大元素误差:        {max_err:.6e}")
    print(f"精度转换用时:        {elapsed * 1e3:.3f} ms")
    print_quantization_bitrate(A, U, s, Vt, r, 8, 8, per_col=None)
    return {"abs_err": abs_err, "rel_err": rel_err, "max_err": max_err, "time_s": elapsed}


# fp4 e2m1 可表示的非负数值(1 符号位 + 2 指数位 + 1 尾数位)
# 含 subnormal 0.5,共 8 个非负层级;CPU 上 torch 原生 fp4 dtype 暂不支持
# 直接 cast,故手动模拟 round-to-nearest。
_FP4_LEVELS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def to_fp4_e2m1(x: torch.Tensor) -> torch.Tensor:
    """将浮点张量按 fp4 e2m1 语义舍入,返回同 dtype 浮点张量。

    e2m1 可表示值: ±{0, 0.5, 1, 1.5, 2, 3, 4, 6}。
    对每个元素取绝对值后舍入到最近层级(中点向上),再恢复符号。
    """
    levels = torch.tensor(_FP4_LEVELS, dtype=x.dtype, device=x.device)
    # 层级中点作为分界: [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0]
    boundaries = (levels[1:] + levels[:-1]) / 2
    sign = torch.sign(x)
    idx = torch.bucketize(x.abs(), boundaries)
    return sign * levels[idx]


def quantize_mxfp4_per_col(x: torch.Tensor) -> torch.Tensor:
    """模拟 MXFP4 量化/反量化,每列沿行维每 32 个元素共享 E8M0 scale。

    scale = 2**max(floor(log2(block_amax)) - 2, -127),无中心偏移。
    E2M1 使用 nearest-even 舍入并饱和到 +/-6;尾块补零后裁回原形状。
    仅接受有限浮点二维输入及可表示的缩放指数 (最大 127)。
    返回原 dtype 的反量化值,不生成打包 FP4 或 E8M0 字节。
    """
    if x.ndim != 2 or not x.is_floating_point():
        raise ValueError("MXFP4 expects a floating-point matrix")
    if not torch.isfinite(x).all():
        raise ValueError("MXFP4 expects finite inputs")
    if x.numel() == 0:
        return x.clone()

    rows, columns = x.shape
    padded = torch.nn.functional.pad(x.T.to(torch.float64), (0, (-rows) % 32))
    blocks = padded.reshape(columns, -1, 32)
    block_amax = blocks.abs().amax(dim=-1, keepdim=True)
    _, max_exponent = torch.frexp(block_amax)
    scale_exponent = (max_exponent - 3).clamp_min(-127)
    scale_exponent = torch.where(block_amax == 0, 0, scale_exponent)
    if torch.any(scale_exponent > 127):
        raise ValueError("MXFP4 scale exponent exceeds the E8M0 range")
    scale = torch.exp2(scale_exponent.to(torch.float64))
    magnitude = (blocks / scale).abs()

    levels = torch.tensor(_FP4_LEVELS, dtype=blocks.dtype, device=x.device)
    boundaries = (levels[1:] + levels[:-1]) / 2
    indices = torch.bucketize(magnitude.contiguous(), boundaries)
    ties = magnitude == boundaries[indices.clamp_max(boundaries.numel() - 1)]
    indices = indices + (ties & (indices % 2 == 1)).to(indices.dtype)
    quantized = torch.copysign(levels[indices], blocks) * scale
    return quantized.reshape(columns, -1)[:, :rows].T.to(x.dtype).contiguous()


def mxfp4_reconstruction_error(
    A: torch.Tensor, U: torch.Tensor, s: torch.Tensor, Vt: torch.Tensor, r: int
) -> dict:
    """U/V 按列分块 MXFP4 模拟量化,保留 s,报告重构误差与量化时间。"""
    U_r = U[:, :r]
    V_r = Vt[:r, :].T
    s_r = s[:r]

    t0 = time.perf_counter()
    U_q = quantize_mxfp4_per_col(U_r)
    V_q = quantize_mxfp4_per_col(V_r)
    elapsed = time.perf_counter() - t0

    A_q = (U_q * s_r) @ V_q.T
    abs_err = torch.linalg.norm(A - A_q, ord="fro").item()
    rel_err = abs_err / torch.linalg.norm(A, ord="fro").item()
    max_err = (A - A_q).abs().max().item()

    print("\n=== MXFP4 按列分块量化重构误差 (block=32, E8M0 scale, U/V 量化, s 保留) ===")
    print(f"绝对误差 (Frobenius): {abs_err:.6e}")
    print(f"相对误差:            {rel_err:.6e}")
    print(f"最大元素误差:        {max_err:.6e}")
    print(f"量化用时:            {elapsed * 1e3:.3f} ms")
    print_quantization_bitrate(A, U, s, Vt, r, 4, 4, mxfp4=True)
    return {"abs_err": abs_err, "rel_err": rel_err, "max_err": max_err, "time_s": elapsed}


def hadamard_mxfp4_reconstruction_error(
    A: torch.Tensor, U: torch.Tensor, s: torch.Tensor, Vt: torch.Tensor, r: int
) -> dict:
    """U/V 每列沿行维 Hadamard 旋转、MXFP4 量化、反旋转后重构。

    U/V 的行数须为 2 的幂。保留 s,计时包含旋转矩阵构造及正反旋转,
    不含最终重构和误差计算。
    """
    U_r = U[:, :r]
    V_r = Vt[:r, :].T
    s_r = s[:r]

    t0 = time.perf_counter()
    hadamard_U = hadamard_matrix(U_r.shape[0], dtype=U_r.dtype, device=U_r.device)
    hadamard_V = hadamard_matrix(V_r.shape[0], dtype=V_r.dtype, device=V_r.device)
    U_rot = hadamard_U @ U_r
    V_rot = hadamard_V @ V_r
    U_q = quantize_mxfp4_per_col(U_rot)
    V_q = quantize_mxfp4_per_col(V_rot)
    U_final = hadamard_U.T @ U_q
    V_final = hadamard_V.T @ V_q
    elapsed = time.perf_counter() - t0

    A_q = (U_final * s_r) @ V_final.T
    abs_err = torch.linalg.norm(A - A_q, ord="fro").item()
    rel_err = abs_err / torch.linalg.norm(A, ord="fro").item()
    max_err = (A - A_q).abs().max().item()

    print("\n=== Hadamard 旋转 + MXFP4 重构误差 (block=32, E8M0 scale, s 保留) ===")
    print(f"绝对误差 (Frobenius): {abs_err:.6e}")
    print(f"相对误差:            {rel_err:.6e}")
    print(f"最大元素误差:        {max_err:.6e}")
    print(f"量化用时:            {elapsed * 1e3:.3f} ms")
    print_quantization_bitrate(A, U, s, Vt, r, 4, 4, mxfp4=True)
    return {"abs_err": abs_err, "rel_err": rel_err, "max_err": max_err, "time_s": elapsed}


def quantize_fp4_asym_per_col(x: torch.Tensor) -> torch.Tensor:
    """非对称 fp4 (e2m1) 按列量化,返回反量化后的浮点张量。

    x: (n, k) — 每列(长度 n)独立计算仿射参数,再舍入到 e2m1 网格。
      center = (max + min) / 2          # 列中心(等效连续 zero-point)
      scale  = (max - min) / 2 / 6      # 半范围映射到码本最大值 ±6
      q      = round_to_e2m1((x - center) / scale)
      deq    = q * scale + center

    与非对称 int4 的关系:同样是利用完整 [min, max] 区间的仿射映射,
    区别在于反量化网格从均匀整数 [-8,7] 换成 e2m1 浮点编码
    ±{0, 0.5, 1, 1.5, 2, 3, 4, 6}(非均匀,大值粒度粗、小值粒度细)。
    scale 归一后元素落在 [-6, 6] 内,解决了裸 fp4 转换中
    幅值 < 0.25 全部舍入到 0 的问题。
    """
    min_val = x.amin(dim=0, keepdim=True)   # (1, k)
    max_val = x.amax(dim=0, keepdim=True)   # (1, k)
    center = (min_val + max_val) / 2
    half_range = (max_val - min_val) / 2
    # 防止常数列除零
    scale = torch.where(half_range > 0, half_range / 6.0, torch.ones_like(half_range))

    q = to_fp4_e2m1((x - center) / scale)
    return q * scale + center


def fp4_reconstruction_error(
    A: torch.Tensor, U: torch.Tensor, s: torch.Tensor, Vt: torch.Tensor, r: int
) -> dict:
    """对 U、V 的前 r 个奇异向量按列做非对称 fp4 (e2m1) 量化后重构,打印误差。

    量化对象:
      - U[:, :r] (n, r): 左奇异向量,按列(长度 n)量化
      - Vt[:r, :].T (n, r): 右奇异向量,按列(长度 n)量化
    奇异值 s 不量化。
    """
    U_r = U[:, :r]            # (n, r)
    V_r = Vt[:r, :].T         # (n, r)
    s_r = s[:r]

    t0 = time.perf_counter()
    U_f4 = quantize_fp4_asym_per_col(U_r)
    V_f4 = quantize_fp4_asym_per_col(V_r)
    elapsed = time.perf_counter() - t0

    A_f4 = U_f4 @ torch.diag(s_r) @ V_f4.T

    abs_err = torch.linalg.norm(A - A_f4, ord="fro").item()
    rel_err = abs_err / torch.linalg.norm(A, ord="fro").item()
    max_err = (A - A_f4).abs().max().item()

    print("\n=== 非对称 fp4 (e2m1) 按列量化重构误差 (U, V 量化, s 保留) ===")
    print(f"绝对误差 (Frobenius): {abs_err:.6e}")
    print(f"相对误差:            {rel_err:.6e}")
    print(f"最大元素误差:        {max_err:.6e}")
    print(f"量化用时:            {elapsed * 1e3:.3f} ms")
    print_quantization_bitrate(A, U, s, Vt, r, 4, 4, per_col="scale_center")
    return {"abs_err": abs_err, "rel_err": rel_err, "max_err": max_err, "time_s": elapsed}


def int4_U_only_reconstruction_error(
    A: torch.Tensor, U: torch.Tensor, s: torch.Tensor, Vt: torch.Tensor, r: int
) -> dict:
    """仅对 U 做非对称 int4 按列量化,V 保持高精度,重构并打印误差。

    用于消融:分离 U 侧与 V 侧量化各自引入的误差。
    重构: A_q = quant(U[:, :r]) @ diag(s[:r]) @ V[:, :r]^T
    """
    U_r = U[:, :r]            # (n, r)
    V_r = Vt[:r, :].T         # (n, r)
    s_r = s[:r]

    t0 = time.perf_counter()
    U_q = quantize_int4_per_col(U_r)
    elapsed = time.perf_counter() - t0
    # V 不量化,直接用原始奇异向量

    A_q = U_q @ torch.diag(s_r) @ V_r.T

    abs_err = torch.linalg.norm(A - A_q, ord="fro").item()
    rel_err = abs_err / torch.linalg.norm(A, ord="fro").item()
    max_err = (A - A_q).abs().max().item()

    print("\n=== 仅 U 非对称 int4 按列量化重构误差 (V 不量化, s 保留) ===")
    print(f"绝对误差 (Frobenius): {abs_err:.6e}")
    print(f"相对误差:            {rel_err:.6e}")
    print(f"最大元素误差:        {max_err:.6e}")
    print(f"量化用时:            {elapsed * 1e3:.3f} ms")
    print_quantization_bitrate(A, U, s, Vt, r, 4, None)
    return {"abs_err": abs_err, "rel_err": rel_err, "max_err": max_err, "time_s": elapsed}


def int4_U_quant_update_V_reconstruction_error(
    A: torch.Tensor, U: torch.Tensor, s: torch.Tensor, Vt: torch.Tensor, r: int
) -> dict:
    """量化 U 后,固定量化 U' 用最小二乘重新求解 V,再用新 U'、V' 重构 A。

    步骤:
      1. U' = 非对称 int4 量化/反量化后的 U[:, :r]
      2. 固定 U',最小二乘求 M = argmin ||A - U' M||_F  (M: r x n)
         等效于在量化后的列空间中找 A 的最优逼近
      3. 由 M 反解新的 V':M = diag(s_r) @ V'.T
         即 V'.T = diag(1/s_r) @ M (V' 不再保证单位正交)
      4. 重构 A_q = U' @ diag(s_r) @ V'.T

    原理:量化 U 引入的误差 ΔU 中,落在 U' 列空间内的分量可被
    重新求解的 V 吸收补偿,因此误差应小于"仅量化 U"(V 不更新)。
    """
    U_r = U[:, :r]            # (n, r)
    s_r = s[:r]

    # 1. 量化/反量化 U
    t0 = time.perf_counter()
    U_q = quantize_int4_per_col(U_r)

    # 2. 固定 U',最小二乘求解 M (r x n)
    M = torch.linalg.lstsq(U_q, A).solution

    # 3. 反解新 V': M = diag(s_r) @ V'.T
    V_new = (M / s_r.unsqueeze(1)).T      # (n, r)
    elapsed = time.perf_counter() - t0

    # 4. 重构
    A_q = U_q @ torch.diag(s_r) @ V_new.T

    abs_err = torch.linalg.norm(A - A_q, ord="fro").item()
    rel_err = abs_err / torch.linalg.norm(A, ord="fro").item()
    max_err = (A - A_q).abs().max().item()

    print("\n=== U 量化 int4 + V 最小二乘更新重构误差 (s 保留) ===")
    print(f"绝对误差 (Frobenius): {abs_err:.6e}")
    print(f"相对误差:            {rel_err:.6e}")
    print(f"最大元素误差:        {max_err:.6e}")
    print(f"量化用时:            {elapsed * 1e3:.3f} ms")
    print_quantization_bitrate(A, U, s, Vt, r, 4, None)
    return {"abs_err": abs_err, "rel_err": rel_err, "max_err": max_err, "time_s": elapsed}


def int4_U_quant_update_V_then_quant_V_reconstruction_error(
    A: torch.Tensor, U: torch.Tensor, s: torch.Tensor, Vt: torch.Tensor, r: int
) -> dict:
    """在"U 量化 + V 最小二乘更新"基础上,再对更新后的 V 做 int4 量化重构。

    步骤:
      1. U' = 非对称 int4 量化/反量化后的 U[:, :r]
      2. 固定 U',最小二乘求 M = argmin ||A - U' M||_F  (M: r x n)
      3. 反解新 V':V'.T = diag(1/s_r) @ M  (V' 不再单位正交)
      4. 对 V' 再做非对称 int4 按列量化,得 V''
      5. 重构 A_q = U' @ diag(s_r) @ V''.T

    与 int4_U_quant_update_V 的区别:最后把最小二乘得到的高精度 V' 也量化,
    使 U、V 两侧均为 int4 存储,评估 V 二次量化引入的额外误差。
    """
    U_r = U[:, :r]            # (n, r)
    s_r = s[:r]

    # 1. 量化/反量化 U
    t0 = time.perf_counter()
    U_q = quantize_int4_per_col(U_r)

    # 2. 固定 U',最小二乘求解 M (r x n)
    M = torch.linalg.lstsq(U_q, A).solution

    # 3. 反解新 V':M = diag(s_r) @ V'.T
    V_new = (M / s_r.unsqueeze(1)).T      # (n, r)

    # 4. 对更新后的 V' 再做 int4 量化
    V_q = quantize_int4_per_col(V_new)
    elapsed = time.perf_counter() - t0

    # 5. 重构
    A_q = U_q @ torch.diag(s_r) @ V_q.T

    abs_err = torch.linalg.norm(A - A_q, ord="fro").item()
    rel_err = abs_err / torch.linalg.norm(A, ord="fro").item()
    max_err = (A - A_q).abs().max().item()

    print("\n=== U 量化 int4 + V 最小二乘更新 + V 再量化 int4 重构误差 (s 保留) ===")
    print(f"绝对误差 (Frobenius): {abs_err:.6e}")
    print(f"相对误差:            {rel_err:.6e}")
    print(f"最大元素误差:        {max_err:.6e}")
    print(f"量化用时:            {elapsed * 1e3:.3f} ms")
    print_quantization_bitrate(A, U, s, Vt, r, 4, 4)
    return {"abs_err": abs_err, "rel_err": rel_err, "max_err": max_err, "time_s": elapsed}


def int4_V_quant_update_U_then_quant_U_reconstruction_error(
    A: torch.Tensor, U: torch.Tensor, s: torch.Tensor, Vt: torch.Tensor, r: int
) -> dict:
    """先量化 V,固定量化 V' 用最小二乘更新 U,再对更新后的 U 量化重构。

    与 int4_U_quant_update_V_then_quant_V 对称(U/V 角色互换):
      1. V' = 非对称 int4 量化/反量化后的 Vt[:r, :].T
      2. 固定 B = diag(s_r) @ V'.T (r x n),最小二乘求 U':
         U' = argmin ||A - U' B||_F  (等效在量化后的行空间中逼近 A)
      3. 对 U' 再做非对称 int4 按列量化,得 U''
      4. 重构 A_q = U'' @ diag(s_r) @ V'.T
    """
    V_r = Vt[:r, :].T         # (n, r)
    s_r = s[:r]

    # 1. 量化/反量化 V
    t0 = time.perf_counter()
    V_q = quantize_int4_per_col(V_r)

    # 2. 固定 B = diag(s_r) @ V_q.T,最小二乘求 U':A = U' B  =>  A.T = B.T U'.T
    B = torch.diag(s_r) @ V_q.T           # (r, n)
    U_new = torch.linalg.lstsq(B.T, A.T).solution.T   # (n, r)

    # 3. 对更新后的 U' 再做 int4 量化
    U_q = quantize_int4_per_col(U_new)
    elapsed = time.perf_counter() - t0

    # 4. 重构
    A_q = U_q @ torch.diag(s_r) @ V_q.T

    abs_err = torch.linalg.norm(A - A_q, ord="fro").item()
    rel_err = abs_err / torch.linalg.norm(A, ord="fro").item()
    max_err = (A - A_q).abs().max().item()

    print("\n=== V 量化 int4 + U 最小二乘更新 + U 再量化 int4 重构误差 (s 保留) ===")
    print(f"绝对误差 (Frobenius): {abs_err:.6e}")
    print(f"相对误差:            {rel_err:.6e}")
    print(f"最大元素误差:        {max_err:.6e}")
    print(f"量化用时:            {elapsed * 1e3:.3f} ms")
    print_quantization_bitrate(A, U, s, Vt, r, 4, 4)
    return {"abs_err": abs_err, "rel_err": rel_err, "max_err": max_err, "time_s": elapsed}


def int4_alternating_quant_UVU_reconstruction_error(
    A: torch.Tensor, U: torch.Tensor, s: torch.Tensor, Vt: torch.Tensor, r: int
) -> dict:
    """交替量化: 量化 U -> 最小二乘更新 V -> 量化 V -> 最小二乘更新 U -> 量化 U 重构。

    在"先量化一侧、最小二乘补偿另一侧"的基础上多走一轮,让最后被量化的 U
    也能吸收 V 量化引入的误差(交替最小化 + 逐步量化):
      1. U1 = int4(U[:, :r])
      2. 固定 U1 最小二乘求 V':V'.T = diag(1/s_r) @ lstsq(U1, A);  V1 = int4(V')
      3. 固定 V1 构造 B = diag(s_r) @ V1.T,最小二乘求 U':A = U' B; U2 = int4(U')
      4. 重构 A_q = U2 @ diag(s_r) @ V1.T
    """
    U_r = U[:, :r]            # (n, r)
    s_r = s[:r]

    # 1. 量化 U
    t0 = time.perf_counter()
    U1 = quantize_int4_per_col(U_r)

    # 2. 固定 U1 更新并量化 V
    M = torch.linalg.lstsq(U1, A).solution        # (r, n)
    V_new = (M / s_r.unsqueeze(1)).T              # (n, r)
    V1 = quantize_int4_per_col(V_new)

    # 3. 固定 V1 更新并量化 U
    B = torch.diag(s_r) @ V1.T                    # (r, n)
    U_new = torch.linalg.lstsq(B.T, A.T).solution.T   # (n, r)
    U2 = quantize_int4_per_col(U_new)
    elapsed = time.perf_counter() - t0

    # 4. 重构
    A_q = U2 @ torch.diag(s_r) @ V1.T

    abs_err = torch.linalg.norm(A - A_q, ord="fro").item()
    rel_err = abs_err / torch.linalg.norm(A, ord="fro").item()
    max_err = (A - A_q).abs().max().item()

    print("\n=== 交替量化 U->更新V->量化V->更新U->量化U 重构误差 (s 保留) ===")
    print(f"绝对误差 (Frobenius): {abs_err:.6e}")
    print(f"相对误差:            {rel_err:.6e}")
    print(f"最大元素误差:        {max_err:.6e}")
    print(f"量化用时:            {elapsed * 1e3:.3f} ms")
    print_quantization_bitrate(A, U, s, Vt, r, 4, 4)
    return {"abs_err": abs_err, "rel_err": rel_err, "max_err": max_err, "time_s": elapsed}


def int4_sequential_ls_reconstruction_error(
    A: torch.Tensor, U: torch.Tensor, s: torch.Tensor, Vt: torch.Tensor, r: int
) -> dict:
    """逐列量化 U、V,每列锁定后最小二乘调整剩余列。"""
    if r < 1:
        raise ValueError("r 必须大于 0")
    U_r = U[:, :r]
    V_r = Vt[:r, :].T
    s_r = s[:r]
    target = (U_r * s_r) @ V_r.T

    t0 = time.perf_counter()
    U_q = U_r.clone()
    for col in range(r):
        U_q[:, col:col + 1] = quantize_int4_per_col(U_q[:, col:col + 1])
        if col + 1 < r:
            residual = target - (U_q[:, :col + 1] * s_r[:col + 1]) @ V_r[:, :col + 1].T
            remaining_V = V_r[:, col + 1:] * s_r[col + 1:]
            U_q[:, col + 1:] = torch.linalg.lstsq(remaining_V, residual.T).solution.T

    V_q = V_r.clone()  # 克隆 V_r 以便逐列量化和调整
    for col in range(r):
        V_q[:, col:col + 1] = quantize_int4_per_col(V_q[:, col:col + 1])
        if col + 1 < r:
            residual = target - (U_q[:, :col + 1] * s_r[:col + 1]) @ V_q[:, :col + 1].T
            remaining_U = U_q[:, col + 1:] * s_r[col + 1:]
            V_q[:, col + 1:] = torch.linalg.lstsq(remaining_U, residual).solution.T
    elapsed = time.perf_counter() - t0

    A_q = (U_q * s_r) @ V_q.T
    abs_err = torch.linalg.norm(A - A_q, ord="fro").item()
    rel_err = abs_err / torch.linalg.norm(A, ord="fro").item()
    max_err = (A - A_q).abs().max().item()

    print("\n=== U/V 逐列 int4 量化 + 剩余列最小二乘调整重构误差 (s 保留) ===")
    print(f"绝对误差 (Frobenius): {abs_err:.6e}")
    print(f"相对误差:            {rel_err:.6e}")
    print(f"最大元素误差:        {max_err:.6e}")
    print(f"量化用时:            {elapsed * 1e3:.3f} ms")
    print_quantization_bitrate(A, U, s, Vt, r, 4, 4)
    return {"abs_err": abs_err, "rel_err": rel_err, "max_err": max_err, "time_s": elapsed}


def _direct_quantization_reconstruction_error(
    A: torch.Tensor, bits: int, use_hadamard: bool = False,
    quantization: str = "int",
) -> dict:
    """完整 A 按列量化,可先左乘 Hadamard;计入相应格式的解码参数。

    无 SVD 或秩截断。Hadamard 要求行数为 2 的幂,列数不限。
    计时包含旋转矩阵构造、正反旋转及量化,不含误差和存储统计。
    整数使用 float32 scale/uint8 zero,k-means 码本按原 dtype 存储;
    MXFP4 使用 E8M0 scale。返回反量化浮点值,不打包整数码值。
    """
    if A.ndim != 2 or not A.is_floating_point() or A.numel() == 0:
        raise ValueError("expected a nonempty floating-point matrix")
    if not torch.isfinite(A).all():
        raise ValueError("expected finite matrix values")
    if quantization not in ("int", "mxfp4", "kmeans"):
        raise ValueError("unsupported direct quantization format")
    if quantization == "int" and bits not in (2, 4, 8):
        raise ValueError("direct integer quantization supports 2, 4 or 8 bits")
    if quantization != "int" and bits != 4:
        raise ValueError("direct MXFP4 and k-means comparisons require 4 bits")
    if use_hadamard and A.shape[0] & (A.shape[0] - 1):
        raise ValueError("Hadamard requires a power-of-two row count")
    quantizers = {4: quantize_int4_per_col, 8: quantize_int8_per_col}

    t0 = time.perf_counter()
    transform = hadamard_matrix(A.shape[0], dtype=A.dtype, device=A.device) if use_hadamard else None
    A_rot = transform @ A if transform is not None else A
    if quantization == "mxfp4":
        A_q = quantize_mxfp4_per_col(A_rot)
    elif quantization == "kmeans":
        A_q = quantize_kmeans_per_col(A_rot, bits=bits)
    elif bits == 2:
        column_bits = torch.full((A.shape[1],), bits, dtype=torch.int64, device=A.device)
        A_q = quantize_per_col_bits(A_rot, column_bits)
    else:
        A_q = quantizers[bits](A_rot)
    if transform is not None:
        A_q = transform.T @ A_q
    elapsed = time.perf_counter() - t0

    difference = A - A_q
    abs_err = torch.linalg.norm(difference, ord="fro").item()
    norm = torch.linalg.norm(A, ord="fro").item()
    rel_err = abs_err / norm if norm > 0 else 0.0
    max_err = difference.abs().max().item()

    payload_bits = A.numel() * bits
    storage_bits = A.element_size() * 8
    padding_bits = 0
    if quantization == "mxfp4":
        blocks = ((A.shape[0] + 31) // 32) * A.shape[1]
        padding_bits = (blocks * 32 - A.numel()) * bits
        parameter_bits = blocks * 8
        metadata = f"E8M0 scale, {blocks} x 8 = {parameter_bits} bit"
        format_name = "MXFP4 (block=32)"
    elif quantization == "kmeans":
        centers_per_column = min(2 ** bits, A.shape[0])
        parameter_bits = A.shape[1] * centers_per_column * storage_bits
        metadata = f"码本, {A.shape[1]} 列 x {centers_per_column} 中心 x {storage_bits} = {parameter_bits} bit"
        format_name = f"k-means int{bits}"
    else:
        parameter_bits = A.shape[1] * (32 + 8)
        metadata = f"scale/zero, {A.shape[1]} x (32 float32 + 8 uint8) = {parameter_bits} bit"
        format_name = f"非对称 int{bits}"
    total_bits = payload_bits + padding_bits + parameter_bits
    rotation = "Hadamard + " if use_hadamard else ""
    print(f"\n=== 完整矩阵 A {rotation}{format_name} 按列量化重构误差 (无 SVD/秩截断) ===")
    print(f"绝对误差 (Frobenius): {abs_err:.6e}")
    print(f"相对误差:            {rel_err:.6e}")
    print(f"最大元素误差:        {max_err:.6e}")
    print(f"量化用时:            {elapsed * 1e3:.3f} ms")
    print(f"A 码值位宽:          {bits} bit/元素")
    print(f"A 按列元数据:        {metadata}")
    print(f"理论总数据量:        {total_bits} bit (A码值={payload_bits}, 元数据={parameter_bits}, 填充={padding_bits}, 无 U/V/s)")
    print(f"理论量化比特率:      {total_bits / A.numel():.6f} bit/A元素 (含解码数据;固定算法/形状/配置共享,不含文件头/对齐)")
    return {"abs_err": abs_err, "rel_err": rel_err, "max_err": max_err, "time_s": elapsed}


def int2_direct_quantization_reconstruction_error(A: torch.Tensor, r: int = 16) -> dict:
    """完整 A 按列 int2 量化;r 仅用于兼容调用。"""
    return _direct_quantization_reconstruction_error(A, bits=2)


def int4_direct_quantization_reconstruction_error(A: torch.Tensor, r: int = 16) -> dict:
    """完整 A 按列 int4 量化;保留 r 兼容对比调用,不参与计算。"""
    return _direct_quantization_reconstruction_error(A, bits=4)


def int8_direct_quantization_reconstruction_error(A: torch.Tensor, r: int = 16) -> dict:
    """完整 A 按列 int8 量化;保留 r 兼容对比调用,不参与计算。"""
    return _direct_quantization_reconstruction_error(A, bits=8)


def hadamard_int2_direct_quantization_reconstruction_error(A: torch.Tensor, r: int = 16) -> dict:
    """完整 A 左乘 Hadamard、按列 int2 量化、反旋转;r 不参与计算。"""
    return _direct_quantization_reconstruction_error(A, bits=2, use_hadamard=True)


def hadamard_int4_direct_quantization_reconstruction_error(A: torch.Tensor, r: int = 16) -> dict:
    """完整 A 左乘 Hadamard、按列 int4 量化、反旋转;r 不参与计算。"""
    return _direct_quantization_reconstruction_error(A, bits=4, use_hadamard=True)


def hadamard_int8_direct_quantization_reconstruction_error(A: torch.Tensor, r: int = 16) -> dict:
    """完整 A 左乘 Hadamard、按列 int8 量化、反旋转;r 不参与计算。"""
    return _direct_quantization_reconstruction_error(A, bits=8, use_hadamard=True)


def mxfp4_direct_quantization_reconstruction_error(A: torch.Tensor, r: int = 16) -> dict:
    """完整 A 按列分块 MXFP4 量化;r 不参与计算。"""
    return _direct_quantization_reconstruction_error(A, bits=4, quantization="mxfp4")


def hadamard_mxfp4_direct_quantization_reconstruction_error(A: torch.Tensor, r: int = 16) -> dict:
    """完整 A 左乘 Hadamard、按列分块 MXFP4 量化、反旋转;r 不参与计算。"""
    return _direct_quantization_reconstruction_error(A, bits=4, use_hadamard=True, quantization="mxfp4")


def kmeans_int4_direct_quantization_reconstruction_error(A: torch.Tensor, r: int = 16) -> dict:
    """完整 A 按列 k-means int4 量化,每列独立码本;r 不参与计算。"""
    return _direct_quantization_reconstruction_error(A, bits=4, quantization="kmeans")


def hadamard_kmeans_int4_direct_quantization_reconstruction_error(A: torch.Tensor, r: int = 16) -> dict:
    """完整 A 左乘 Hadamard、按列 k-means int4 量化、反旋转;r 不参与计算。"""
    return _direct_quantization_reconstruction_error(A, bits=4, use_hadamard=True, quantization="kmeans")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="低秩矩阵 SVD 量化重构误差验证。默认用随机低秩矩阵,"
                    "指定 --from-dump 时按 layer/head/step 读取真实 ssm_state 矩阵。"
    )
    parser.add_argument("--from-dump", action="store_true",
                        help="从 dump 读取矩阵而非构造随机矩阵。")
    parser.add_argument("--dump-root", default=DEFAULT_DUMP_ROOT,
                        help="dump 根目录 (默认: %(default)s)。")
    parser.add_argument("--layer", type=int, default=0, help="层索引 (默认 0)。")
    parser.add_argument("--head", type=int, default=0, help="head 索引 (默认 0)。")
    parser.add_argument("--step", type=int, default=None,
                        help="step 编号,默认取最后一个 step。")
    parser.add_argument("--r", type=int, default=16, help="重构使用的秩 r (默认 16)。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    r = args.r

    if args.from_dump:
        A = load_matrix_from_dump(args.dump_root, args.layer, args.head, args.step)
    else:
        n = 128
        # 构造低秩矩阵(近似低秩，不完全16秩)，故意射着r=n，用于模拟实际场景
        A = build_low_rank_matrix(n=n, r=n)

    rank = torch.linalg.matrix_rank(A).item()
    print(f"原始矩阵 shape={tuple(A.shape)}, 数值秩={rank}")

    U, s, Vt = torch.linalg.svd(A, full_matrices=False)

    energy_rank(s)
    analyze_uv_distribution(U, Vt, r=r)
    baseline_reconstruction_error(A, U, s, Vt, r=r)
    int4_quantized_reconstruction_error(A, U, s, Vt, r=r)
    # gptq_int4_reconstruction_error(A, U, s, Vt, r=r)
    # uniform_clip_sigma_int4_reconstruction_error(A, U, s, Vt, r=r, clip_sigmas=2.5)
    # for bits in (1, 2, 3):
    #     lloyd_max_intN_reconstruction_error(A, U, s, Vt, r=r, bits=bits)
    #     uniform_intN_reconstruction_error(A, U, s, Vt, r=r, bits=bits)
    # lloyd_max_intN_reconstruction_error(A, U, s, Vt, r=r, bits=4)
    lloyd_max_int4_reconstruction_error(A, U, s, Vt, r=r)
    kmeans_intN_reconstruction_error(A, U, s, Vt, r=r, bits=4)
    # kmeans_per_tensor_reconstruction_error(A, U, s, Vt, r=r, bits=4)
    companding_reconstruction_error(A, U, s, Vt, r=r, bits=4, clip_sigmas=2.5)
    # for bits in (5, 6, 7, 8):
    #     lloyd_max_intN_reconstruction_error(A, U, s, Vt, r=r, bits=bits)
    # int5_quantized_reconstruction_error(A, U, s, Vt, r=r)
    int6_quantized_reconstruction_error(A, U, s, Vt, r=r)
    # int7_quantized_reconstruction_error(A, U, s, Vt, r=r)
    int8_quantized_reconstruction_error(A, U, s, Vt, r=r)
    mixed_precision_by_singular_value_reconstruction_error(A, U, s, Vt, r=r, avg_bits=4.0)
    mixed_precision_U_quant_update_V_then_quant_V_reconstruction_error(A, U, s, Vt, r=r, avg_bits=4.0)
    mixed_precision_alternating_quant_UVU_reconstruction_error(A, U, s, Vt, r=r, avg_bits=4.0)
    hadamard_int4_quantized_reconstruction_error(A, U, s, Vt, r=r)
    # fp8_reconstruction_error(A, U, s, Vt, r=r)
    # fp4_reconstruction_error(A, U, s, Vt, r=r)
    mxfp4_reconstruction_error(A, U, s, Vt, r=r)
    hadamard_mxfp4_reconstruction_error(A, U, s, Vt, r=r)
    # int4_U_only_reconstruction_error(A, U, s, Vt, r=r)
    # int4_U_quant_update_V_reconstruction_error(A, U, s, Vt, r=r)
    # int4_U_quant_update_V_then_quant_V_reconstruction_error(A, U, s, Vt, r=r)
    # int4_V_quant_update_U_then_quant_U_reconstruction_error(A, U, s, Vt, r=r)
    int4_alternating_quant_UVU_reconstruction_error(A, U, s, Vt, r=r)
    int4_sequential_ls_reconstruction_error(A, U, s, Vt, r=r)

    # 不对U/V进行量化的重构误差，而是直接量化整个矩阵A
    int2_direct_quantization_reconstruction_error(A, r=r)
    int4_direct_quantization_reconstruction_error(A, r=r)
    int8_direct_quantization_reconstruction_error(A, r=r)
    hadamard_int2_direct_quantization_reconstruction_error(A, r=r)
    hadamard_int4_direct_quantization_reconstruction_error(A, r=r)
    hadamard_int8_direct_quantization_reconstruction_error(A, r=r)
    mxfp4_direct_quantization_reconstruction_error(A, r=r)
    hadamard_mxfp4_direct_quantization_reconstruction_error(A, r=r)
    kmeans_int4_direct_quantization_reconstruction_error(A, r=r)
    hadamard_kmeans_int4_direct_quantization_reconstruction_error(A, r=r)


if __name__ == "__main__":
    main()
