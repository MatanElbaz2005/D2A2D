import numpy as np
try:
    from creedsolo import RSCodec, ReedSolomonError
except ImportError:
    print("cant find creedsolo, using reedsolo instead")
    from reedsolo import RSCodec, ReedSolomonError
import time
from protected_jpeg import merge_jpeg, fix_false_markers
from helpers_files.helpers import decode_codewords, _decode_len_block_chips_dataonly
import binxcorr
import yaml
from helpers_files.config_helpers import _cfg, get_rsc, get_marker_codebook, get_headers_sync, get_prbs
from helpers_files.runtime_helpers import _rt_print

def decode_frame_to_udp(frame: np.ndarray, _RUNTIME=None, HEADER_TEMPLATE_READY=False, HEADER_TEMPLATE=None, corr_threshold: float = 0.9) -> bytes:
    cfg = _cfg()
    FRAME_WIDTH  = cfg["frame"]["width"]
    FRAME_HEIGHT = cfg["frame"]["height"]

    USE_RS_FOR_DATA = cfg["rs"]["use_for_data"]
    CHUNK_BYTES     = cfg["rs"]["chunk_bytes"]
    ECC_SYMBOLS     = cfg["rs"]["ecc_symbols"]

    USE_MARKER_CODEWORDS = cfg["markers"]["use"]
    MARKER_CODEWORD_LEN  = cfg["markers"]["codeword_len"]
    MARKER_DET_THRESH    = cfg["markers"]["det_thresh"]

    USE_PRBS_FOR_HEADERS  = cfg["prbs"]["use_for_headers"]
    USE_PRBS_FOR_DATA     = cfg["prbs"]["use_for_data"]
    CHIP_LENGTH_FOR_DATA  = cfg["prbs"]["chip_length_for_data"]
    DATA_PRBS_POLY        = cfg["prbs"]["data_prbs_poly"]

    LENGTH_BITS_PER_FIELD = cfg["length"]["bits_per_field"]
    LENGTH_CHIP_LENGTH    = cfg["length"]["chip_length"]

    HEADERS_SYNC_PATTERN = get_headers_sync()
    LENGTH_PRBS = get_prbs(LENGTH_CHIP_LENGTH, tuple(DATA_PRBS_POLY), seed=3) if USE_PRBS_FOR_HEADERS else None
    DATA_PRBS   = get_prbs(CHIP_LENGTH_FOR_DATA, tuple(DATA_PRBS_POLY),   seed=3) if USE_PRBS_FOR_DATA    else None

    _TOKENS, _CODES = get_marker_codebook()
    _CODES_PACKED = np.packbits((_CODES > 0).astype(np.uint8), axis=1)
    rsc = get_rsc()               

    if frame.shape != (FRAME_HEIGHT, FRAME_WIDTH):
        raise ValueError(f"Frame size mismatch: expected {FRAME_HEIGHT}x{FRAME_WIDTH}")
    t0 = time.time()
    received_pm = (2 * (frame.ravel() > 127).astype(np.int8) - 1)
    t1 = time.time()
    _rt_print(_RUNTIME, "[DEC] Threshold->±1 took: ", t1 - t0)

    t = time.time()
    search_end = max(len(HEADERS_SYNC_PATTERN), int(received_pm.size * 0.10))
    h_corr, h_idx = binxcorr.correlate_sliding_bin_argmax(received_pm, HEADERS_SYNC_PATTERN, 0, search_end, False)
    _rt_print(_RUNTIME, "[DEC] 1×correlate (sync only): ", time.time() - t, "s")

    if h_corr < corr_threshold:
        raise ValueError(f"Sync not detected (headers): {h_corr:.3f}")

    sync_start = int(h_idx)
    sync_end   = sync_start + len(HEADERS_SYNC_PATTERN)

    len_bits_total = LENGTH_BITS_PER_FIELD
    chips_per_bit  = (LENGTH_CHIP_LENGTH if USE_PRBS_FOR_HEADERS else 3)
    len_block_chips = len_bits_total * chips_per_bit

    len_block_rx = received_pm[sync_end: sync_end + len_block_chips]
    data_chips_len = _decode_len_block_chips_dataonly(
        len_block_rx, USE_PRBS_FOR_HEADERS, LENGTH_CHIP_LENGTH, LENGTH_PRBS if USE_PRBS_FOR_HEADERS else None, LENGTH_BITS_PER_FIELD
    )

    data_start = sync_end + len_block_chips
    data_end   = data_start + data_chips_len
    if not (data_start < data_end):
        raise ValueError(f"Invalid data range: data_start={data_start}, data_end={data_end}")

    if not HEADER_TEMPLATE_READY or HEADER_TEMPLATE is None:
        raise ValueError("HEADER_TEMPLATE not initialised")
    decoded_headers = HEADER_TEMPLATE

    
    # Despread data
    protected_data    = received_pm[data_start:data_end]
    if USE_MARKER_CODEWORDS:
        t = time.time()
        data_dec = decode_codewords(protected_data.astype(np.int8, copy=False),_TOKENS, _CODES_PACKED, MARKER_CODEWORD_LEN, MARKER_DET_THRESH,return_token_positions=True)
        if isinstance(data_dec, tuple):
            data_bytes, token_starts = data_dec
        else:
            data_bytes, token_starts = data_dec, None
        _rt_print(_RUNTIME, "[DEC] Marker codewords decode took: ", time.time() - t)
    else:
        t = time.time()
        if USE_PRBS_FOR_DATA:
            n_groups_data = len(protected_data) // CHIP_LENGTH_FOR_DATA
            chips_data = protected_data[:n_groups_data * CHIP_LENGTH_FOR_DATA].reshape(-1, CHIP_LENGTH_FOR_DATA)
            rx_bits_pm_data = np.dot(chips_data, DATA_PRBS) / CHIP_LENGTH_FOR_DATA
            rx_bits_data = ((np.sign(rx_bits_pm_data) + 1) / 2).astype(np.uint8)
        else:
            rx_bits_data = ((protected_data + 1) / 2).astype(np.uint8)
        data_bytes = np.packbits(rx_bits_data).tobytes()
        _rt_print(_RUNTIME, "[DEC] Despread/map+packbits data took: ", time.time() - t)

    
    if USE_RS_FOR_DATA:
        t_rs_data = time.time()
        decoded_chunks = []
        i = 0
        N = CHUNK_BYTES + ECC_SYMBOLS
        while i + N <= len(data_bytes):
            blk = data_bytes[i:i+N]
            try:
                decoded_chunks.append(bytes(rsc.decode(bytearray(blk))[0]))
            except ReedSolomonError:
                # replace with black/zero payload for this chunk
                decoded_chunks.append(blk[:CHUNK_BYTES])
            i += N
        # last (possibly shorter) block
        if i < len(data_bytes):
            blk = data_bytes[i:]
            try:
                decoded_chunks.append(bytes(rsc.decode(bytearray(blk))[0]))
            except ReedSolomonError:
                k_last = max(0, len(blk) - ECC_SYMBOLS)
                decoded_chunks.append(blk[:k_last])
        decoded_data = b"".join(decoded_chunks)
        _rt_print(_RUNTIME, "[DEC] RS data (chunked) took: ", time.time() - t_rs_data)
    else:
        decoded_data = data_bytes
    
    t = time.time()
    if USE_RS_FOR_DATA:
        fixed_data = fix_false_markers(
            decoded_data,
            use_marker_codewords=False,
            preserve_data=True,
            whitelist_rst=None
        )
    else:
        fixed_data = fix_false_markers(
            decoded_data,
            use_marker_codewords=USE_MARKER_CODEWORDS,
            preserve_data=True,
            whitelist_rst=token_starts
        )
    _rt_print(_RUNTIME, "[DEC] fix_false_markers took: ", time.time() - t)

    t7 = time.time()
    result = merge_jpeg(decoded_headers, fixed_data)
    _rt_print(_RUNTIME, "[DEC] merge_jpeg took: ", time.time() - t7)
    return result