import simsimd as simd 
import numpy as np

def _pack_pm_to_bytes(pm_int8: np.ndarray) -> np.ndarray:
    bits01 = (pm_int8 > 0).astype(np.uint8, copy=False)
    return np.packbits(bits01, bitorder="big")

def _byte_shift_view(b: np.ndarray, s: int) -> np.ndarray:
    assert b.dtype == np.uint8
    if s == 0:
        return b
    if b.size < 2:
        return b[:0]
    hi = (b[:-1] >> s).astype(np.uint8, copy=False)
    lo = ((b[1:] << (8 - s)) & 0xFF).astype(np.uint8, copy=False)
    return (hi | lo).astype(np.uint8, copy=False)

def _correlate_binary_full(received_pm: np.ndarray,
                           sync_pm: np.ndarray,
                           sync_pack: np.ndarray,
                           last_mask: np.uint8) -> np.ndarray:
    N = int(received_pm.size)
    L = int(sync_pm.size)
    if N < L:
        return np.empty(0, dtype=np.float32)

    B = int(sync_pack.size)
    src_bytes = _pack_pm_to_bytes(received_pm)
    sB = src_bytes.strides[0]
    total_pos = N - L + 1
    corr = np.empty(total_pos, dtype=np.float32)

    for s in range(8):
        vb = _byte_shift_view(src_bytes, s)
        m = vb.size - B + 1
        if m <= 0:
            continue

        windows = np.lib.stride_tricks.as_strided(
            vb, shape=(m, B), strides=(sB, sB), writeable=False
        )

        if last_mask != 0xFF:
            win = windows.copy()
            win[:, -1] &= last_mask
        else:
            win = windows

        # SIMD Hamming over packed bytes (bit-level)
        ham = simd.hamming(win, sync_pack, dtype="bin8").astype(np.int32, copy=False)

        corr_part = (1.0 - 2.0 * (ham.astype(np.float32) / float(L)))

        exp_len = ((total_pos - 1 - s) // 8) + 1 if (total_pos - 1) >= s else 0
        use = min(m, exp_len)
        if use > 0:
            corr[s : s + 8*use : 8] = corr_part[:use]

    return corr

def binary_sync_correlate_roi(received_pm: np.ndarray, HEADERS_SYNC_PATTERN, _HDR_PACK, _HDR_MASK, DATA_SYNC_PATTERN, _DAT_PACK, _DAT_MASK, END_SYNC_PATTERN, _END_PACK, _END_MASK) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    ch = _correlate_binary_full(received_pm, HEADERS_SYNC_PATTERN, _HDR_PACK, _HDR_MASK)
    cd = _correlate_binary_full(received_pm, DATA_SYNC_PATTERN,    _DAT_PACK, _DAT_MASK)
    data_start = int(np.argmax(cd)) + DATA_SYNC_PATTERN.size
    ce_roi = _correlate_binary_full(received_pm[data_start:], END_SYNC_PATTERN, _END_PACK, _END_MASK)
    return ch, cd, ce_roi, data_start
