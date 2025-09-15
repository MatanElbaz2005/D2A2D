import numpy as np
from helpers_files.helpers import _encode_len_block_chips_dataonly, _encode_data_with_codewords_fast
from helpers_files.runtime_helpers import _rt_print
import time
from helpers_files.config_helpers import _cfg, get_rsc, get_marker_codebook, get_headers_sync, get_prbs


def encode_udp_to_frame_dataonly(data: bytes, *, _RUNTIME: dict) -> tuple[np.ndarray, dict]:
    cfg = _cfg()

    FRAME_WIDTH  = cfg["frame"]["width"]
    FRAME_HEIGHT = cfg["frame"]["height"]

    # RS
    USE_RS_FOR_DATA = cfg["rs"]["use_for_data"]
    CHUNK_BYTES     = cfg["rs"]["chunk_bytes"]
    rsc = get_rsc()

    # Markers
    USE_MARKER_CODEWORDS = cfg["markers"]["use"]
    TOKENS, CODES = get_marker_codebook()

    # Sync
    HEADERS_SYNC_PATTERN = get_headers_sync()

    # Length
    LENGTH_BITS_PER_FIELD = cfg["length"]["bits_per_field"]
    USE_PRBS_FOR_HEADERS  = cfg["prbs"]["use_for_headers"]
    LENGTH_CHIP_LENGTH    = cfg["length"]["chip_length"]
    DATA_PRBS_POLY        = cfg["prbs"]["data_prbs_poly"]
    LENGTH_PRBS = get_prbs(LENGTH_CHIP_LENGTH, tuple(DATA_PRBS_POLY), seed=3) if USE_PRBS_FOR_HEADERS else None

    # Data PRBS
    USE_PRBS_FOR_DATA   = cfg["prbs"]["use_for_data"]
    CHIP_LENGTH_FOR_DATA = cfg["prbs"]["chip_length_for_data"]
    DATA_PRBS = get_prbs(CHIP_LENGTH_FOR_DATA, tuple(DATA_PRBS_POLY), seed=3) if USE_PRBS_FOR_DATA else None

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

# def encode_udp_to_frame(headers: bytes, data: bytes) -> tuple[np.ndarray, dict]:
#     t_outer0 = time.time()
    
#     if USE_RS_FOR_HEADERS:
#         start_rs_headers_encode = time.time()
#         coded_headers = rsc.encode(bytearray(headers))
#         _rt_print(_RUNTIME, "[ENC] rs encode for headers took ", time.time() - start_rs_headers_encode)
#     else:
#         coded_headers = headers
    
#     t = time.time()
#     header_bits = np.unpackbits(np.frombuffer(coded_headers, dtype=np.uint8))
#     header_bits_pm = header_bits.astype(np.int8) * 2 - 1
#     _rt_print(_RUNTIME, "[ENC] headers unpack->pm:", time.time() - t)

#     if USE_PRBS_FOR_HEADERS:
#         repeated_bits = np.repeat(header_bits_pm, CHIP_LENGTH_FOR_HEADERS)
#         tiled_prbs = np.tile(HEADERS_PRBS, len(header_bits))
#         protected_headers = repeated_bits * tiled_prbs
#     else:
#         mapping = {0: np.array([-1, 1, -1]), 1: np.array([1, -1, 1])}
#         protected_headers = np.concatenate([mapping[bit] for bit in header_bits])
#     _rt_print(_RUNTIME, "[ENC] PRBS/map headers: ", time.time() - t, "s")
    
#     if USE_RS_FOR_DATA:
#         start_rs_data_encode = time.time()
#         coded_blocks = []
#         for i in range(0, len(data), CHUNK_BYTES):
#             blk = data[i:i+CHUNK_BYTES]
#             coded_blocks.append(rsc.encode(bytearray(blk))) 
#         coded_data = b"".join(coded_blocks)
#         _rt_print(_RUNTIME, "[ENC] rs encode (chunked) took ", time.time() - start_rs_data_encode)
#     else:
#         coded_data = data

    
#     if USE_MARKER_CODEWORDS:
#         t = time.time()
#         protected_data = _encode_data_with_codewords_fast(coded_data, _TOKENS, _CODES)
#         _rt_print(_RUNTIME, "[ENC] Marker codewords encode: ", time.time() - t, "s")
#     else:
#         t = time.time()
#         data_bits = np.unpackbits(np.frombuffer(coded_data, dtype=np.uint8))
#         data_bits_pm = data_bits.astype(np.int8) * 2 - 1
#         _rt_print(_RUNTIME, "[ENC] data unpack->pm:", time.time() - t, "s")

#         t = time.time()
#         if USE_PRBS_FOR_DATA:
#             repeated_data_bits = np.repeat(data_bits_pm, CHIP_LENGTH_FOR_DATA)
#             tiled_prbs = np.tile(DATA_PRBS, len(data_bits))
#             protected_data = repeated_data_bits * tiled_prbs
#         else:
#             protected_data = data_bits_pm
#         _rt_print(_RUNTIME, "[ENC] PRBS/map data: ", time.time() - t, "s")

#     t = time.time()
#     len_block_pm = _encode_len_block_chips(
#         hdr_len_chips=len(protected_headers),
#         data_len_chips=len(protected_data),
#         use_prbs=USE_PRBS_FOR_HEADERS,
#         chip_len=LENGTH_CHIP_LENGTH,
#         prbs=LENGTH_PRBS if USE_PRBS_FOR_HEADERS else None,
#         LENGTH_BITS_PER_FIELD=LENGTH_BITS_PER_FIELD
#     )
#     _rt_print(_RUNTIME, "[ENC] build length block:", time.time() - t, " s")
#     t = time.time()
#     full_stream = np.concatenate((HEADERS_SYNC_PATTERN, len_block_pm, protected_headers, protected_data))
#     _rt_print(_RUNTIME, "[ENC] Concat full stream: ", time.time()-t, "s len=", len(full_stream))

#     t = time.time()
#     s0 = 0
#     s1 = s0 + len(HEADERS_SYNC_PATTERN)
#     s2 = s1 + len(len_block_pm)
#     s3 = s2 + len(protected_headers)
#     s4 = s3 + len(protected_data)

#     tx_meta = {
#         "stream_pm": full_stream.astype(np.int8),
#         "idx": {
#             "sync": (s0, s1),
#             "len":  (s1, s2),
#             "hdr":  (s2, s3),
#             "data": (s3, s4),
#         }
#     }
#     _rt_print(_RUNTIME, "[ENC] build tx_meta:", time.time() - t, " s")
#     _rt_print(_RUNTIME, "[ENC] Full stream length: ", len(full_stream), " bits")

#     t = time.time()
#     total_pixels = FRAME_WIDTH * FRAME_HEIGHT
#     if len(full_stream) > total_pixels:
#         raise ValueError(f"Data too large: {len(full_stream)} bits > {total_pixels} pixels")
    
#     full_u8 = (((full_stream + 1) // 2).astype(np.uint8)) * 255
#     if full_u8.size < total_pixels:
#         full_u8 = np.pad(full_u8, (0, total_pixels - full_u8.size), mode='constant')
#     frame = full_u8.reshape((FRAME_HEIGHT, FRAME_WIDTH))
#     _rt_print(_RUNTIME, "[ENC] pack/pad/reshape frame:", time.time() - t, " s")

#     return frame, tx_meta
