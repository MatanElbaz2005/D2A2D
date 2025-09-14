import numpy as np
from helpers_files.helpers import _encode_len_block_chips_dataonly, _encode_data_with_codewords_fast
from helpers_files.runtime_helpers import _rt_print
import time


def encode_udp_to_frame_dataonly(
    data: bytes,
    *,
    USE_RS_FOR_DATA: bool,
    CHUNK_BYTES: int,
    rsc,
    USE_MARKER_CODEWORDS: bool,
    TOKENS,
    CODES,
    USE_PRBS_FOR_DATA: bool,
    CHIP_LENGTH_FOR_DATA: int,
    DATA_PRBS,
    USE_PRBS_FOR_HEADERS: bool,
    LENGTH_CHIP_LENGTH: int,
    LENGTH_PRBS,
    LENGTH_BITS_PER_FIELD: int,
    HEADERS_SYNC_PATTERN: np.ndarray,
    FRAME_WIDTH: int,
    FRAME_HEIGHT: int,
    _RUNTIME: dict
) -> tuple[np.ndarray, dict]:
    if USE_RS_FOR_DATA:
        start_rs_data_encode = time.time()
        coded_blocks = []
        for i in range(0, len(data), CHUNK_BYTES):
            blk = data[i:i+CHUNK_BYTES]
            coded_blocks.append(rsc.encode(bytearray(blk)))
        coded_data = b"".join(coded_blocks)
        _rt_print(_RUNTIME, "[ENC] rs encode (chunked) took ", time.time() - start_rs_data_encode)
    else:
        coded_data = data

    if USE_MARKER_CODEWORDS:
        t = time.time()
        protected_data = _encode_data_with_codewords_fast(coded_data, TOKENS, CODES)
        _rt_print(_RUNTIME, "[ENC] Marker codewords encode: ", time.time() - t, "s")
    else:
        t = time.time()
        data_bits = np.unpackbits(np.frombuffer(coded_data, dtype=np.uint8))
        data_bits_pm = data_bits.astype(np.int8) * 2 - 1
        _rt_print(_RUNTIME, "[ENC] data unpack->pm:", time.time() - t, "s")

        t = time.time()
        if USE_PRBS_FOR_DATA:
            repeated = np.repeat(data_bits_pm, CHIP_LENGTH_FOR_DATA)
            tiled = np.tile(DATA_PRBS, len(data_bits))
            protected_data = repeated * tiled
        else:
            protected_data = data_bits_pm
        _rt_print(_RUNTIME, "[ENC] PRBS/map data: ", time.time() - t, "s")

    t = time.time()
    len_block_pm = _encode_len_block_chips_dataonly(
        data_len_chips=len(protected_data),
        use_prbs=USE_PRBS_FOR_HEADERS,
        chip_len=LENGTH_CHIP_LENGTH,
        prbs=LENGTH_PRBS if USE_PRBS_FOR_HEADERS else None,
        LENGTH_BITS_PER_FIELD=LENGTH_BITS_PER_FIELD
    )
    _rt_print(_RUNTIME, "[ENC] build length block:", time.time() - t, " s")

    t = time.time()
    full_stream = np.concatenate((HEADERS_SYNC_PATTERN, len_block_pm, protected_data)).astype(np.int8)
    _rt_print(_RUNTIME, "[ENC] Concat full stream: ", time.time()-t, "s len=", len(full_stream))

    t = time.time()
    s0 = 0
    s1 = s0 + len(HEADERS_SYNC_PATTERN)
    s2 = s1 + len(len_block_pm)
    s3 = s2 + len(protected_data)

    tx_meta = {"stream_pm": full_stream,
               "idx": {"sync": (s0, s1), "len": (s1, s2), "data": (s2, s3)}}       
    _rt_print(_RUNTIME, "[ENC] build tx_meta:", time.time() - t, " s")
    _rt_print(_RUNTIME, "[ENC] Full stream length: ", len(full_stream), " bits")

    t = time.time()
    total_pixels = FRAME_WIDTH * FRAME_HEIGHT
    if full_stream.size > total_pixels:
        raise ValueError(f"Data too large: {full_stream.size} bits > {total_pixels} pixels")
    full_u8 = (((full_stream + 1)//2).astype(np.uint8))*255
    if full_u8.size < total_pixels:
        full_u8 = np.pad(full_u8, (0, total_pixels - full_u8.size), mode='constant')
    frame = full_u8.reshape((FRAME_HEIGHT, FRAME_WIDTH))
    _rt_print(_RUNTIME, "[ENC] pack/pad/reshape frame:", time.time() - t, " s")

    return frame, tx_meta
