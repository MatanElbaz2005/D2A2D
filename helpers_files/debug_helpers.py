import numpy as np
import cv2
from typing import Optional, Set, Tuple, Dict, List
from statistics import mean, pstdev

from protected_jpeg import fix_false_markers, merge_jpeg

def _ssim_gray(img1: np.ndarray, img2: np.ndarray) -> float:
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = kernel @ kernel.T
    mu1 = cv2.filter2D(img1, -1, window)
    mu2 = cv2.filter2D(img2, -1, window)
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1*img1, -1, window) - mu1_sq
    sigma2_sq = cv2.filter2D(img2*img2, -1, window) - mu2_sq
    sigma12   = cv2.filter2D(img1*img2, -1, window) - mu1_mu2
    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2)) / ((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
    return float(ssim_map.mean())

def ssim_color(imgA_bgr: np.ndarray, imgB_bgr: np.ndarray) -> float:
    yA = cv2.cvtColor(imgA_bgr, cv2.COLOR_BGR2YCrCb)[:,:,0]
    yB = cv2.cvtColor(imgB_bgr, cv2.COLOR_BGR2YCrCb)[:,:,0]
    return _ssim_gray(yA, yB)

def rebuild_two_variants(
    decoded_headers: bytes,
    data_bytes: bytes,
    use_marker_codewords: bool,
    token_starts: Optional[Set[int]],
) -> Tuple[Tuple[bytes, Dict[str, int]], Tuple[bytes, Dict[str, int]]]:
    stats_ff00: Dict[str, int] = {}
    fixed_ff00 = fix_false_markers(
        data_bytes,
        use_marker_codewords=use_marker_codewords,
        preserve_data=False,
        whitelist_rst=token_starts,
        stats=stats_ff00
    )
    jpg_ff00 = merge_jpeg(decoded_headers, fixed_ff00)

    stats_efxx: Dict[str, int] = {}
    fixed_efxx = fix_false_markers(
        data_bytes,
        use_marker_codewords=use_marker_codewords,
        preserve_data=True,
        whitelist_rst=token_starts,
        stats=stats_efxx
    )
    jpg_efxx = merge_jpeg(decoded_headers, fixed_efxx)

    return (jpg_ff00, stats_ff00), (jpg_efxx, stats_efxx)

def accumulate_stats(total: Dict[str, int], frame_stats: Dict[str, int], prefix: str):
    """Accumulate counters into a flat dict with a given prefix (e.g., 'ff_' or 'ef_')."""
    for k, v in frame_stats.items():
        total[f"{prefix}{k}"] = total.get(f"{prefix}{k}", 0) + int(v)

def summarize_metrics(records: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """
    records: list of dicts per-frame with keys like: psnr_ff, ssim_ff, psnr_ef, ssim_ef, sigma, etc.
    returns: {'psnr_ff': {'mean':..., 'std':...}, ...}
    """
    if not records:
        return {}
    keys = [k for k in records[0].keys() if isinstance(records[0][k], (int, float))]
    out: Dict[str, Dict[str, float]] = {}
    for k in keys:
        vals = [float(r.get(k, float('nan'))) for r in records if r.get(k) is not None]
        if not vals:
            continue
        out[k] = {
            "averge": float(mean(vals)),
            "std":  float(pstdev(vals)) if len(vals) > 1 else 0.0
        }
    return out

def hstack3(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Simple horizontal stack of 3 same-size BGR images."""
    return cv2.hconcat([a, b, c])

def make_info_bar(width: int, text: str, height: int = 36) -> np.ndarray:
    bar = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.rectangle(bar, (0, 0), (width - 1, height - 1), (30, 30, 30), -1)
    cv2.putText(bar, text, (14, height - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(bar, text, (14, height - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return bar

def build_summary_json(records: List[Dict[str, float]], totals: Dict[str, int], frames: int, elapsed_sec: float) -> Dict:
    agg = summarize_metrics(records)
    summary = {
        "meta": {
            "frames": frames,
            "sec": elapsed_sec
        },
        "metrics": {
            "FF00": {
                "psnr": {"averge": agg.get("psnr_ff", {}).get("averge", None), "std": agg.get("psnr_ff", {}).get("std", None)},
                "ssim": {"averge": agg.get("ssim_ff", {}).get("averge", None), "std": agg.get("ssim_ff", {}).get("std", None)},
            },
            "EFxx": {
                "psnr": {"averge": agg.get("psnr_ef", {}).get("averge", None), "std": agg.get("psnr_ef", {}).get("std", None)},
                "ssim": {"averge": agg.get("ssim_ef", {}).get("averge", None), "std": agg.get("ssim_ef", {}).get("std", None)},
            },
        },
        "totals": {
            "FF00": {
                "false_rst": totals.get("ff_false_rst", 0),
                "illegal_ffxx": totals.get("ff_illegal_ffxx", 0),
                "to_ff00": totals.get("ff_to_ff00", 0),
                "to_efxx": totals.get("ff_to_efxx", 0),
            },
            "EFxx": {
                "false_rst": totals.get("ef_false_rst", 0),
                "illegal_ffxx": totals.get("ef_illegal_ffxx", 0),
                "to_ff00": totals.get("ef_to_ff00", 0),
                "to_efxx": totals.get("ef_to_efxx", 0),
            },
        }
    }
    return summary

