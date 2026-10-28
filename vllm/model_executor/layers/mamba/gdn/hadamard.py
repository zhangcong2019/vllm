from __future__ import annotations

import math

import torch

_LLOYD_MAX_INT4_LEVELS_STD = [
    -2.7344, -2.0688, -1.6178, -1.2552, -0.9413, -0.6564, -0.3881, -0.1280,
    +0.1278, +0.3872, +0.6558, +0.9421, +1.2556, +1.6189, +2.0750, +2.7430,
]
_LLOYD_MAX_INT2_LEVELS_STD = [
    -1.5100, -0.4527, +0.4527, +1.5100,
]
_LLOYD_MAX_INT1_LEVELS_STD = [
    -0.7976, +0.7976,
]

def normalized_fwht(x: torch.Tensor) -> torch.Tensor:
    """Apply the orthonormal Walsh-Hadamard transform over the final dimension."""
    if not torch.is_floating_point(x):
        raise TypeError("normalized_fwht requires a floating-point tensor")
    if x.ndim < 1:
        raise ValueError("normalized_fwht requires at least one dimension")

    n = x.shape[-1]
    if n == 0 or n & (n - 1):
        raise ValueError("the final dimension must be a nonzero power of two")

    result = x.clone()
    h = 1
    while h < n:
        blocks = result.reshape(*result.shape[:-1], n // (2 * h), 2, h)
        left = blocks[..., 0, :].clone()
        right = blocks[..., 1, :].clone()
        result = torch.stack((left + right, left - right), dim=-2).reshape_as(result)
        h *= 2

    return result / math.sqrt(n)


def inverse_normalized_fwht(x: torch.Tensor) -> torch.Tensor:
    """Invert an orthonormal Walsh-Hadamard transform over the final dimension."""
    return normalized_fwht(x)


def normalized_fwht_pertensor(x: torch.Tensor) -> torch.Tensor:
    """Apply the orthonormal Walsh-Hadamard transform over the final dimension."""
    if not torch.is_floating_point(x):
        raise TypeError("normalized_fwht requires a floating-point tensor")
    if x.ndim < 1:
        raise ValueError("normalized_fwht requires at least one dimension")

    original_shape = x.shape
    flat = x.reshape(-1)

    n = flat.shape[-1]
    if n == 0 or n & (n - 1):
        raise ValueError("the total number of elements must be a nonzero power of two")

    result = flat.clone()
    h = 1
    while h < n:
        blocks = result.reshape(*result.shape[:-1], n // (2 * h), 2, h)
        left = blocks[..., 0, :].clone()
        right = blocks[..., 1, :].clone()
        result = torch.stack((left + right, left - right), dim=-2).reshape_as(result)
        h *= 2

    result = result / math.sqrt(n)

    result = result.reshape(original_shape)

    return result


def inverse_normalized_fwht_pertensor(x: torch.Tensor) -> torch.Tensor:
    """Invert an orthonormal Walsh-Hadamard transform over the final dimension."""
    return normalized_fwht_pertensor(x)


def asymmetric_int4_quantize(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize each final-dimension vector to asymmetric signed INT4 values."""
    if not torch.is_floating_point(x):
        raise TypeError("asymmetric_int4_quantize requires a floating-point tensor")
    if x.ndim < 1:
        raise ValueError("asymmetric_int4_quantize requires at least one dimension")
    if x.shape[-1] == 0:
        raise ValueError("the final dimension must be nonempty")

    minimum = x.amin(dim=-1, keepdim=True)
    maximum = x.amax(dim=-1, keepdim=True)
    raw_scale = (maximum - minimum) / 15
    constant = raw_scale == 0
    scale = torch.where(constant, torch.ones_like(raw_scale), raw_scale)
    zero_point = torch.where(constant, minimum, minimum + 8 * scale)
    q = torch.round((x - zero_point) / scale).clamp(-8, 7).to(torch.int8)
    return q, scale, zero_point


def asymmetric_int4_dequantize(
    q: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor
) -> torch.Tensor:
    """Dequantize signed INT4 values stored in an INT8 tensor."""
    if q.dtype != torch.int8:
        raise TypeError("asymmetric_int4_dequantize requires an INT8 quantized tensor")

    return q * scale + zero_point

def asymmetric_int2_quantize(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize each final-dimension vector to asymmetric signed INT2 values."""
    if not torch.is_floating_point(x):
        raise TypeError("asymmetric_int2_quantize requires a floating-point tensor")
    if x.ndim < 1:
        raise ValueError("asymmetric_int2_quantize requires at least one dimension")
    if x.shape[-1] == 0:
        raise ValueError("the final dimension must be nonempty")

    minimum = x.amin(dim=-1, keepdim=True)
    maximum = x.amax(dim=-1, keepdim=True)
    raw_scale = (maximum - minimum) / 3
    constant = raw_scale == 0
    scale = torch.where(constant, torch.ones_like(raw_scale), raw_scale)
    zero_point = torch.where(constant, minimum, minimum + 2 * scale)
    q = torch.round((x - zero_point) / scale).clamp(-2, 1).to(torch.int8)
    return q, scale, zero_point


def asymmetric_int2_dequantize(
    q: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor
) -> torch.Tensor:
    """Dequantize signed INT2 values stored in an INT8 tensor."""
    if q.dtype != torch.int8:
        raise TypeError("asymmetric_int2_dequantize requires an INT8 quantized tensor")

    return q * scale + zero_point


def asymmetric_int1_quantize(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize each final-dimension vector to asymmetric signed INT1 values."""
    if not torch.is_floating_point(x):
        raise TypeError("asymmetric_int1_quantize requires a floating-point tensor")
    if x.ndim < 1:
        raise ValueError("asymmetric_int1_quantize requires at least one dimension")
    if x.shape[-1] == 0:
        raise ValueError("the final dimension must be nonempty")

    minimum = x.amin(dim=-1, keepdim=True)
    maximum = x.amax(dim=-1, keepdim=True)
    raw_scale = maximum - minimum
    constant = raw_scale == 0
    scale = torch.where(constant, torch.ones_like(raw_scale), raw_scale)
    zero_point = torch.where(constant, minimum, minimum + scale)
    q = torch.round((x - zero_point) / scale).clamp(-1, 0).to(torch.int8)
    print("###asymmetric_int1_quantize: scale.shape =", scale.shape)
    return q, scale, zero_point


def asymmetric_int1_dequantize(
    q: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor
) -> torch.Tensor:
    """Dequantize signed INT1 values stored in an INT8 tensor."""
    if q.dtype != torch.int8:
        raise TypeError("asymmetric_int1_dequantize requires an INT8 quantized tensor")

    return q * scale + zero_point


def kmeans_int2_quantize(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize each final-dimension vector using offline Lloyd-Max levels.

    Normalize each vector to unit L2 norm so that, assuming zero mean, its
    entries have a standard deviation of 1 / sqrt(x.shape[-1]). Scale the
    standard-normal levels without running online k-means. Return signed INT2
    labels, a four-entry codebook of unit-norm centroid values, and the
    per-vector L2 norm as scale; use kmeans_int2_dequantize to reconstruct values.
    """
    if not torch.is_floating_point(x):
        raise TypeError("kmeans_int2_quantize requires a floating-point tensor")
    if x.ndim < 1:
        raise ValueError("kmeans_int2_quantize requires at least one dimension")
    if x.shape[-1] == 0:
        raise ValueError("the final dimension must be nonempty")

    values = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
    scale = values.norm(dim=-1, keepdim=True)
    zero = scale == 0
    normalized = values / torch.where(zero, torch.ones_like(scale), scale)

    levels = values.new_tensor(_LLOYD_MAX_INT2_LEVELS_STD)
    std = x.shape[-1] ** -0.5
    centroids = (std * levels).expand(*x.shape[:-1], 4)

    labels = (normalized.unsqueeze(-1) - centroids.unsqueeze(-2)).abs().argmin(-1)
    q = (labels - 2).to(torch.int8)
    return q, centroids, scale


def kmeans_int2_dequantize(
    q: torch.Tensor, codebook: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    """Reconstruct -2/-1/0/1 labels using the scaled offline Lloyd-Max codebook.

    The four levels are nonuniform, so reconstruct by lookup rather than an
    affine mapping, then restore the per-vector L2 norm recorded as scale.
    """
    if q.dtype != torch.int8:
        raise TypeError("kmeans_int2_dequantize requires an INT8 quantized tensor")
    if torch.any((q < -2) | (q > 1)):
        raise ValueError("kmeans_int2_dequantize requires labels in {-2, -1, 0, 1}")

    return codebook.gather(-1, q.long() + 2) * scale


def turbo_int2_quantize(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply normalized FWHT, then quantize each final-dimension vector."""
    return kmeans_int2_quantize(normalized_fwht(x))

def turbo_int2_dequantize(
    q: torch.Tensor, codebook: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    """Reconstruct signed INT2 labels using their per-vector codebooks."""
    if q.dtype != torch.int8:
        raise TypeError("turbo_int2_dequantize requires an INT8 quantized tensor")

    return inverse_normalized_fwht(kmeans_int2_dequantize(q, codebook, scale))


def kmeans_int1_quantize(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize each final-dimension vector using offline Lloyd-Max levels.

    Normalize each vector to unit L2 norm so that, assuming zero mean, its
    entries have a standard deviation of 1 / sqrt(x.shape[-1]). Scale the
    standard-normal levels without running online k-means. Return signed INT1
    labels, a two-entry codebook of unit-norm centroid values, and the per-vector
    L2 norm as scale; use kmeans_int1_dequantize to reconstruct values.
    """
    if not torch.is_floating_point(x):
        raise TypeError("kmeans_int1_quantize requires a floating-point tensor")
    if x.ndim < 1:
        raise ValueError("kmeans_int1_quantize requires at least one dimension")
    if x.shape[-1] == 0:
        raise ValueError("the final dimension must be nonempty")

    values = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
    scale = values.norm(dim=-1, keepdim=True)
    zero = scale == 0
    normalized = values / torch.where(zero, torch.ones_like(scale), scale)

    levels = values.new_tensor(_LLOYD_MAX_INT1_LEVELS_STD)
    std = x.shape[-1] ** -0.5
    centroids = (std * levels).expand(*x.shape[:-1], 2)

    labels = (normalized.unsqueeze(-1) - centroids.unsqueeze(-2)).abs().argmin(-1)
    q = (labels - 1).to(torch.int8)
    return q, centroids, scale


def kmeans_int1_dequantize(
    q: torch.Tensor, codebook: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    """Reconstruct -1/0 labels using the scaled offline Lloyd-Max codebook.

    Map -1 to the lower level and 0 to the upper level directly, then restore
    the per-vector L2 norm recorded as scale.
    """
    if q.dtype != torch.int8:
        raise TypeError("kmeans_int1_dequantize requires an INT8 quantized tensor")
    if torch.any((q != -1) & (q != 0)):
        raise ValueError("kmeans_int1_dequantize requires labels in {-1, 0}")

    lower = codebook[..., :1]
    upper = codebook[..., 1:]
    return (q.to(codebook.dtype) * (upper - lower) + upper) * scale


def turbo_int1_quantize(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply normalized FWHT, then quantize each final-dimension vector."""
    return kmeans_int1_quantize(normalized_fwht(x))

def turbo_int1_dequantize(
    q: torch.Tensor, codebook: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    """Reconstruct signed INT1 labels using their per-vector codebooks."""
    if q.dtype != torch.int8:
        raise TypeError("turbo_int1_dequantize requires an INT8 quantized tensor")

    return inverse_normalized_fwht(kmeans_int1_dequantize(q, codebook, scale))

def hadamard_int2_quantize(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply normalized FWHT, then quantize each final-dimension vector."""
    return asymmetric_int2_quantize(normalized_fwht(x))


def hadamard_int2_dequantize(
    q: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor
) -> torch.Tensor:
    """Dequantize signed values, then apply the inverse normalized FWHT."""
    transformed = asymmetric_int2_dequantize(q, scale, zero_point)
    return inverse_normalized_fwht(transformed)

def hadamard_int1_quantize(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply normalized FWHT, then quantize each final-dimension vector."""
    return asymmetric_int1_quantize(normalized_fwht(x))


def hadamard_int1_dequantize(
    q: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor
) -> torch.Tensor:
    """Dequantize signed values, then apply the inverse normalized FWHT."""
    transformed = asymmetric_int1_dequantize(q, scale, zero_point)
    return inverse_normalized_fwht(transformed)

def hadamard_int4_quantize(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply normalized FWHT, then quantize each final-dimension vector."""
    return asymmetric_int4_quantize(normalized_fwht(x))


def hadamard_int4_dequantize(
    q: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor
) -> torch.Tensor:
    """Dequantize signed values, then apply the inverse normalized FWHT."""
    transformed = asymmetric_int4_dequantize(q, scale, zero_point)
    return inverse_normalized_fwht(transformed)

if __name__ == "__main__":
    # 测试代码    # 生成随机输入：batch_size=4，每个样本长度为8（2的幂）
    torch.manual_seed(42)  # 固定随机种子，便于复现
    x = torch.randn(4, 1024, dtype=torch.float32)  # 也可以使用均匀分布或任意值

    print("原始输入 x:")
    print(x)

    # 量化
    q, scale, zero_point = hadamard_int2_quantize(x)

    # 反量化
    x_reconstructed = hadamard_int2_dequantize(q, scale, zero_point)

    # x_reconstructed = inverse_normalized_fwht(normalized_fwht(x))  # 这里仅示例，实际应使用 hadamard_int4_dequantize

    print("\n反量化后的重建 x_reconstructed:")
    print(x_reconstructed)

    # 比较误差
    diff = x - x_reconstructed
    mse = torch.mean(diff ** 2).item()
    mae = torch.mean(torch.abs(diff)).item()
    max_abs_err = torch.max(torch.abs(diff)).item()

    print("\n误差统计:")
    print(f"均方误差 (MSE) : {mse:.6f}")
    print(f"平均绝对误差 (MAE) : {mae:.6f}")
    print(f"最大绝对误差 : {max_abs_err:.6f}")

    # 也可以逐元素查看差异（前几个）
    print("\n前5个元素的误差 (逐样本):")
    print(diff[0, :5])