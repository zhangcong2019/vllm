import torch

from . import hadamard
from .quantization import mixed_precision_companding_update_V_reconstruction

_compression_rate_total = 0.0
_compression_rate_count = 0
_retained_rank_histogram: dict[int, int] = {}


def _record_retained_rank(retained_rank: int) -> None:
    _retained_rank_histogram[retained_rank] = (
        _retained_rank_histogram.get(retained_rank, 0) + 1
    )
    print(f"retained_rank_histogram={dict(sorted(_retained_rank_histogram.items()))}")


def _record_compression_rate(name: str, compression_rate: float) -> None:
    global _compression_rate_count, _compression_rate_total
    _compression_rate_total += float(compression_rate)
    _compression_rate_count += 1
    average_compression_rate = _compression_rate_total / _compression_rate_count
    print(
        f"{name} compression_rate={compression_rate:.4g} "
        f"average_compression_rate={average_compression_rate:.4g} "
        f"count={_compression_rate_count}"
    )


def _print_frobenius_error(source: torch.Tensor, reconstructed: torch.Tensor) -> None:
    reference_norm = torch.linalg.vector_norm(source.float())
    error_norm = torch.linalg.vector_norm(source.float() - reconstructed.float())
    if reference_norm.item() == 0:
        error_percent = 0.0 if error_norm.item() == 0 else float("inf")
    else:
        error_percent = (100 * error_norm / reference_norm).item()
    print(f"frobenius_error={error_percent:.4g}%, absolute error={error_norm.item():.4g}")

DEFAULT_APPROX_RANK_ENERGY_THRESHOLD = 0.99
DEFAULT_AVG_BITS = 4

def reconstruct_qwen(selected: torch.Tensor) -> torch.Tensor:
    print("Reconstructing Qwen tensor")
    
    # print("Returning random tensor")
    # return torch.nn.functional.normalize(torch.randn_like(selected), dim=-1)

    # print("Returning all-zero tensor")
    # return torch.zeros_like(selected)

    # print("Returning original tensor")
    # return selected

    # print("Returning RTN int2 tensor")
    # return reconstruct_int2_quant(selected)

    # print("Returning hadamard-int2 tensor")
    # return reconstruct_hadamard_int2_quant(selected)

    # print("Returning turbo-int2 tensor")
    # return reconstruct_turbo_int2_quant(selected)

    print("Returning RTN int1 tensor")
    return reconstruct_int1_quant(selected)

    print("Returning hadamard-int1 tensor")
    return reconstruct_hadamard_int1_quant(selected)

    print("Returning turbo-int1 tensor")
    return reconstruct_turbo_int1_quant(selected)

    # print("Returning svd + quant tensor, avg_bits=", DEFAULT_AVG_BITS, "rank_energy_threshold=", DEFAULT_APPROX_RANK_ENERGY_THRESHOLD)
    # return reconstruct_svd_quant(selected)

    # print("Returning svd 099 fp8 tensor")
    # return reconstruct_svd_fp8(selected)

    # print("Returning int4 tensor")
    # return reconstruct_uint4_quant(selected)

def reconstruct_kimi_head(selected: torch.Tensor) -> torch.Tensor:
    print("Reconstructing Kimi tensor")

    # print("Returning random tensor")
    # return torch.nn.functional.normalize(torch.randn_like(selected), dim=-1)

    # print("Returning all-zero tensor")
    # return torch.zeros_like(selected)

    # print("Returning original tensor")
    # return selected

    # print("Returning hadamard-int2 tensor")
    # return reconstruct_hadamard_int2_quant(selected)

    print("Returning turbo-int2 tensor")
    return reconstruct_turbo_int2_quant(selected)

    # print("Returning RTN int2 tensor")
    # return reconstruct_int2_quant(selected)

    # print("Returning svd + quant tensor, avg_bits=", DEFAULT_AVG_BITS, "rank_energy_threshold=", DEFAULT_APPROX_RANK_ENERGY_THRESHOLD)
    # return reconstruct_svd_quant(selected)

    # print("Returning svd 099 fp8 tensor")
    # return reconstruct_svd_fp8(selected)

    # print("Returning int4 tensor")
    # return reconstruct_uint4_quant(selected)
def reconstruct_kimi(selected: torch.Tensor) -> torch.Tensor:
    output = selected.clone()
    for batch_idx in range(selected.shape[0]):
        for head_idx in range(selected.shape[1]):
            output[batch_idx, head_idx] = reconstruct_kimi_head(selected[batch_idx, head_idx])
    return output

def reconstruct_hadamard_int2_quant(selected: torch.Tensor) -> torch.Tensor:
    q, scale, zero_point = hadamard.hadamard_int2_quantize(selected)
    reconstructed = hadamard.hadamard_int2_dequantize(q, scale, zero_point)
    selected.copy_(reconstructed.to(dtype=selected.dtype))
    _record_compression_rate("hadamard", 2.0 / 32.0)
    _print_frobenius_error(selected.float(), reconstructed)
    return selected

def reconstruct_hadamard_int1_quant(selected: torch.Tensor) -> torch.Tensor:
    q, scale, zero_point = hadamard.hadamard_int1_quantize(selected)
    reconstructed = hadamard.hadamard_int1_dequantize(q, scale, zero_point)
    selected.copy_(reconstructed.to(dtype=selected.dtype))
    _record_compression_rate("hadamard-int1", 1.0 / 32.0)
    _print_frobenius_error(selected.float(), reconstructed)
    return selected

def reconstruct_turbo_int2_quant(selected: torch.Tensor) -> torch.Tensor:
    q, centroids = hadamard.turbo_int2_quantize(selected)
    reconstructed = hadamard.turbo_int2_dequantize(q, centroids)
    selected.copy_(reconstructed.to(dtype=selected.dtype))
    _record_compression_rate("turbo", 2.0 / 32.0)
    _print_frobenius_error(selected.float(), reconstructed)
    return selected

def reconstruct_turbo_int1_quant(selected: torch.Tensor) -> torch.Tensor:
    q, centroids = hadamard.turbo_int1_quantize(selected)
    reconstructed = hadamard.turbo_int1_dequantize(q, centroids)
    selected.copy_(reconstructed.to(dtype=selected.dtype))
    _record_compression_rate("turbo-int1", 1.0 / 32.0)
    _print_frobenius_error(selected.float(), reconstructed)
    return selected

def reconstruct_fp8_quant(selected: torch.Tensor) -> torch.Tensor:
    source = selected.float()
    fp8_dtype = torch.float8_e4m3fn
    finfo = torch.finfo(fp8_dtype)
    amax = source.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    scale = finfo.max / amax
    fp8 = (source * scale).clamp(finfo.min, finfo.max).to(fp8_dtype)
    reconstructed = fp8.float() / scale

    _record_compression_rate("fp8", 8.0 / 32.0)
    _print_frobenius_error(source, reconstructed)
    selected.copy_(reconstructed.to(dtype=selected.dtype))
    return selected


def reconstruct_int8_quant(selected: torch.Tensor) -> torch.Tensor:
    source = selected.float()
    int8_info = torch.iinfo(torch.int8)
    x_min = source.amin(dim=-1, keepdim=True)
    x_max = source.amax(dim=-1, keepdim=True)
    scale = ((x_max - x_min) / float(int8_info.max - int8_info.min)).clamp(min=1e-12)
    zero_point = (int8_info.min - x_min / scale).round().clamp(
        int8_info.min, int8_info.max
    )
    q_int8 = (
        (source / scale + zero_point)
        .round()
        .clamp(int8_info.min, int8_info.max)
        .to(torch.int8)
    )
    reconstructed = (q_int8.float() - zero_point) * scale

    _record_compression_rate("int8", 8.0 / 32.0)
    _print_frobenius_error(source, reconstructed)
    selected.copy_(reconstructed.to(dtype=selected.dtype))
    return selected


def reconstruct_uint4_quant(selected: torch.Tensor) -> torch.Tensor:
    source = selected.float()
    qmin, qmax = 0.0, 15.0
    x_min = source.amin(dim=-1, keepdim=True)
    x_max = source.amax(dim=-1, keepdim=True)
    scale = ((x_max - x_min) / (qmax - qmin)).clamp(min=1e-12)
    zero_point = (qmin - x_min / scale).round().clamp(qmin, qmax)
    q_uint4 = (source / scale + zero_point).round().clamp(qmin, qmax)
    reconstructed = (q_uint4 - zero_point) * scale

    _record_compression_rate("uint4", 4.0 / 32.0)
    _print_frobenius_error(source, reconstructed)
    # selected.copy_(reconstructed.to(dtype=selected.dtype))
    return selected


def reconstruct_int2_quant(selected: torch.Tensor) -> torch.Tensor:
    source = selected.float()
    qmin, qmax = 0.0, 3.0
    x_min = source.amin(dim=-1, keepdim=True)
    x_max = source.amax(dim=-1, keepdim=True)
    scale = ((x_max - x_min) / (qmax - qmin)).clamp(min=1e-12)
    zero_point = (qmin - x_min / scale).round().clamp(qmin, qmax)
    q_int2 = (source / scale + zero_point).round().clamp(qmin, qmax)
    reconstructed = (q_int2 - zero_point) * scale

    _record_compression_rate("int2", 2.0 / 32.0)
    _print_frobenius_error(source, reconstructed)
    # selected.copy_(reconstructed.to(dtype=selected.dtype))
    return selected


def reconstruct_int1_quant(selected: torch.Tensor) -> torch.Tensor:
    source = selected.float()
    qmin, qmax = 0.0, 1.0
    x_min = source.amin(dim=-1, keepdim=True)
    x_max = source.amax(dim=-1, keepdim=True)
    scale = ((x_max - x_min) / (qmax - qmin)).clamp(min=1e-12)
    # Keep zero_point unrounded: with only 2 levels, rounding it would force
    # 0 to be representable and collapse one level onto 0.
    zero_point = qmin - x_min / scale
    q_int1 = (source / scale + zero_point).round().clamp(qmin, qmax)
    reconstructed = (q_int1 - zero_point) * scale

    _record_compression_rate("int1", 1.0 / 32.0)
    _print_frobenius_error(source, reconstructed)
    selected.copy_(reconstructed.to(dtype=selected.dtype))
    return selected


def reconstruct_svd_quant(selected: torch.Tensor) -> torch.Tensor:
    approx_rank_energy_threshold = DEFAULT_APPROX_RANK_ENERGY_THRESHOLD
    source = selected.float()

    if source.ndim != 2:
        raise SystemExit(
            "SVD roundtrip expects a 2D matrix after head selection, found shape "
            f"{list(source.shape)}"
        )

    u, singular_values, vh = torch.linalg.svd(source, full_matrices=False)
    if singular_values.numel() == 0:
        reconstructed = source.clone()
        retained_rank = 0
    else:
        energy = singular_values.square()
        total_energy = energy.sum()
        if float(total_energy.item()) == 0.0:
            retained_rank = 0
            reconstructed = torch.zeros_like(source)
        else:
            cumulative_energy = torch.cumsum(energy, dim=0) / total_energy
            retained_rank = (
                int(
                    torch.searchsorted(
                        cumulative_energy, approx_rank_energy_threshold
                    ).item()
                )
                + 1
            )
            u_q = u[:, :retained_rank].to(source.dtype)
            vh_q = vh[:retained_rank, :].to(source.dtype)

            reconstructed_no_quant = (
                u_q
                @ torch.diag(singular_values[:retained_rank].to(torch.float16).to(source.dtype))
                @ vh_q
            )

            reconstructed = mixed_precision_companding_update_V_reconstruction(source, u_q, singular_values[:retained_rank], vh_q, retained_rank, avg_bits=DEFAULT_AVG_BITS)
            # if float(retained_rank) > 20.5:
            #     reconstructed = reconstructed_no_quant

    _record_retained_rank(retained_rank)
    rows, cols = source.shape
    source_bits = rows * cols * 32
    compressed_bits = retained_rank * rows * DEFAULT_AVG_BITS + retained_rank * cols * DEFAULT_AVG_BITS + retained_rank * 16 + retained_rank * 16 * 2 * 2
    compression_rate = compressed_bits / source_bits
    print("average bit", compressed_bits / (rows * cols))
    _record_compression_rate("svd", compression_rate)
    
    print("Frobenius error for non-quantized reconstruction:", end=" ")
    _print_frobenius_error(source, reconstructed_no_quant)
    print("Frobenius error for quantized reconstruction:", end=" ")
    _print_frobenius_error(source, reconstructed)
    print("retained rank:", retained_rank)

    selected.copy_(reconstructed.to(dtype=selected.dtype))
    return selected



def reconstruct_svd_fp8(selected: torch.Tensor) -> torch.Tensor:
    approx_rank_energy_threshold = 0.99
    source = selected.float()

    if source.ndim != 2:
        raise SystemExit(
            "SVD roundtrip expects a 2D matrix after head selection, found shape "
            f"{list(source.shape)}"
        )

    u, singular_values, vh = torch.linalg.svd(source, full_matrices=False)
    if singular_values.numel() == 0:
        reconstructed = source.clone()
        retained_rank = 0
    else:
        energy = singular_values.square()
        total_energy = energy.sum()
        if float(total_energy.item()) == 0.0:
            retained_rank = 0
            reconstructed = torch.zeros_like(source)
        else:
            cumulative_energy = torch.cumsum(energy, dim=0) / total_energy
            retained_rank = (
                int(
                    torch.searchsorted(
                        cumulative_energy, approx_rank_energy_threshold
                    ).item()
                )
                + 1
            )
            
            u_q = u[:, :retained_rank].to(torch.float8_e4m3fn).to(source.dtype)
            vh_q = vh[:retained_rank, :].to(torch.float8_e4m3fn).to(source.dtype)
            reconstructed = (
                u_q
                @ torch.diag(singular_values[:retained_rank].to(torch.float16).to(source.dtype))
                @ vh_q
            )

    _record_retained_rank(retained_rank)
    rows, cols = source.shape
    source_bits = rows * cols * 32
    compressed_bits = retained_rank * rows * 8 + retained_rank * cols * 8 + retained_rank * 16
    compression_rate = compressed_bits / source_bits
    _record_compression_rate("svd", compression_rate)
    _print_frobenius_error(source, reconstructed)
    print("retained rank:", retained_rank)

    selected.copy_(reconstructed.to(dtype=selected.dtype))
    return selected