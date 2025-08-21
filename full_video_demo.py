import numpy as np
try:
    from creedsolo import RSCodec, ReedSolomonError
except ImportError:
    print("cant find creedsolo, using reedsolo instead")
    from reedsolo import RSCodec, ReedSolomonError
from scipy import signal
import cv2
import time
from protected_jpeg import split_jpeg, merge_jpeg, fix_false_markers
from helpers import generate_prbs, _mseq_127_taps_7_1, _mseq_127_taps_7_3, gold127, _decode_data_with_codewords_popcnt
from helpers import _build_marker_codewords_gold, _is_marker_token_at, _encode_data_with_codewords_fast, _decode_data_with_codewords_fast
from gui_helpers import _to_bgr, _compose_grid, _label

# Config
FRAME_WIDTH = 720
FRAME_HEIGHT = 480  # NTSC
ECC_SYMBOLS = 50
CHUNK_BYTES = 150
CHIP_LENGTH_FOR_HEADERS = 3
CHIP_LENGTH_FOR_DATA = 1
DATA_PRBS_POLY = [8, 2]
FORMAT_IMAGE = 'jpg'
MEMORY = [6]
G_MATRIX = [[0o133, 0o171]]
TB_LENGTH = 15
PATH_TO_VIDEO = r"/home/matan/Documents/matan/D2A2D/1572378-sd_960_540_24fps.mp4"
GAUSS_NOISE = 50.0

USE_MARKER_CODEWORDS = True
MARKER_CODEWORD_LEN = 64
MARKER_DET_THRESH   = 0.80

if USE_MARKER_CODEWORDS:
    codewords_time = time.time()
    MARKER_TOKENS = [bytes([0xFF, 0xD0 + i]) for i in range(8)] + [b"\xFF\x00"]
    _TOKENS, _CODES = _build_marker_codewords_gold(MARKER_CODEWORD_LEN, MARKER_TOKENS)
    _CODES_PACKED = np.packbits((_CODES > 0).astype(np.uint8), axis=1)
    means = _CODES.mean(axis=1)

    norm = (_CODES @ _CODES.T) / _CODES.shape[1]
    for i in range(norm.shape[0]):
        norm[i, i] = 0.0

    print(f"[codewords] L={MARKER_CODEWORD_LEN} (token -> codeword 01)")
    for tok, cw in zip(_TOKENS, _CODES):
        s01 = ''.join('1' if int(v) > 0 else '0' for v in cw.tolist())
    print("[ENC] Build marker codewords took: " + str(time.time() - codewords_time))

# PRBS flags
USE_PRBS_FOR_HEADERS = True
USE_PRBS_FOR_DATA = False

# RS flags
USE_RS_FOR_HEADERS = True
USE_RS_FOR_DATA = True

perp_rsc_time = time.time()
rsc = RSCodec(ECC_SYMBOLS)
print("[ENC] preper RS took " + str(time.time() - perp_rsc_time))

# Sync patterns (gold codes, ±1)
HEADERS_SYNC_PATTERN = gold127(shift=0)
DATA_SYNC_PATTERN    = gold127(shift=17)
END_SYNC_PATTERN     = gold127(shift=53)

# PRBS for spreading (if enabled)
prbs_headers_time = time.time()
HEADERS_PRBS = generate_prbs(CHIP_LENGTH_FOR_HEADERS, DATA_PRBS_POLY, 3) if USE_PRBS_FOR_HEADERS else None
if USE_PRBS_FOR_HEADERS: print("[ENC] generate PRBS headers took: " + str(time.time() - prbs_headers_time))
prbs_data_time = time.time()
DATA_PRBS = generate_prbs(CHIP_LENGTH_FOR_DATA, DATA_PRBS_POLY, 3) if USE_PRBS_FOR_DATA else None
if USE_PRBS_FOR_DATA: print("[ENC] generate PRBS data took: " + str(time.time() - prbs_data_time))

def encode_udp_to_frame(headers: bytes, data: bytes) -> tuple[np.ndarray, dict]:
    start_time = time.time()
    
    if USE_RS_FOR_HEADERS:
        start_rs_headers_encode = time.time()
        coded_headers = rsc.encode(bytearray(headers))
        print("[ENC] rs encode for headers took " + str(time.time() - start_rs_headers_encode))
    else:
        coded_headers = headers
    
    t = time.time()
    header_bits = np.unpackbits(np.frombuffer(coded_headers, dtype=np.uint8))
    header_bits_pm = header_bits.astype(np.int8) * 2 - 1

    if USE_PRBS_FOR_HEADERS:
        repeated_bits = np.repeat(header_bits_pm, CHIP_LENGTH_FOR_HEADERS)
        tiled_prbs = np.tile(HEADERS_PRBS, len(header_bits))
        protected_headers = repeated_bits * tiled_prbs
    else:
        mapping = {0: np.array([-1, 1, -1]), 1: np.array([1, -1, 1])}
        protected_headers = np.concatenate([mapping[bit] for bit in header_bits])
    print(f"[ENC] PRBS/map headers: {time.time()-t}s")
    
    if USE_RS_FOR_DATA:
        start_rs_data_encode = time.time()
        coded_blocks = []
        for i in range(0, len(data), CHUNK_BYTES):
            blk = data[i:i+CHUNK_BYTES]
            coded_blocks.append(rsc.encode(bytearray(blk))) 
        coded_data = b"".join(coded_blocks)
        print("[ENC] rs encode (chunked) took " + str(time.time() - start_rs_data_encode))
    else:
        coded_data = data

    
    if USE_MARKER_CODEWORDS:
        t = time.time()
        protected_data = _encode_data_with_codewords_fast(coded_data, _TOKENS, _CODES)
        print(f"[ENC] Marker codewords encode: {time.time()-t}s")
    else:
        t = time.time()
        data_bits = np.unpackbits(np.frombuffer(coded_data, dtype=np.uint8))
        data_bits_pm = data_bits.astype(np.int8) * 2 - 1
        if USE_PRBS_FOR_DATA:
            repeated_data_bits = np.repeat(data_bits_pm, CHIP_LENGTH_FOR_DATA)
            tiled_prbs = np.tile(DATA_PRBS, len(data_bits))
            protected_data = repeated_data_bits * tiled_prbs
        else:
            protected_data = data_bits_pm
        print(f"[ENC] PRBS/map data: {time.time()-t}s")

    t = time.time()
    full_stream = np.concatenate((HEADERS_SYNC_PATTERN, protected_headers, DATA_SYNC_PATTERN, protected_data, END_SYNC_PATTERN))
    print(f"[ENC] Concat full stream: {time.time()-t}s (len={len(full_stream)})")

    s0 = 0
    s1 = s0 + len(HEADERS_SYNC_PATTERN)
    s2 = s1 + len(protected_headers)
    s3 = s2 + len(DATA_SYNC_PATTERN)
    s4 = s3 + len(protected_data)
    s5 = s4 + len(END_SYNC_PATTERN)

    tx_meta = {
        "stream_pm": full_stream.astype(np.int8),
        "idx": {
            "sync_h": (s0, s1),
            "hdr":    (s1, s2),
            "sync_d": (s2, s3),
            "data":   (s3, s4),
            "sync_e": (s4, s5),
        }
    }
    print(f"[ENC] Full stream length: {len(full_stream)} bits")
    
    total_pixels = FRAME_WIDTH * FRAME_HEIGHT
    if len(full_stream) > total_pixels:
        raise ValueError(f"Data too large: {len(full_stream)} bits > {total_pixels} pixels")
    
    full_u8 = (((full_stream + 1) // 2).astype(np.uint8)) * 255
    if full_u8.size < total_pixels:
        full_u8 = np.pad(full_u8, (0, total_pixels - full_u8.size), mode='constant')
    frame = full_u8.reshape((FRAME_HEIGHT, FRAME_WIDTH))
    
    print(f"[ENC] took: {time.time() - start_time} sec")
    return frame, tx_meta

def decode_frame_to_udp(frame: np.ndarray, corr_threshold: float = 0.9) -> bytes:
    if frame.shape != (FRAME_HEIGHT, FRAME_WIDTH):
        raise ValueError(f"Frame size mismatch: expected {FRAME_HEIGHT}x{FRAME_WIDTH}")
    t0 = time.time()
    received_pm = (2 * (frame.ravel() > 127).astype(np.int8) - 1)
    t1 = time.time()
    print("[DEC] Threshold->±1 took: " + str(t1-t0))
    
    t = time.time()
    corr_headers = signal.correlate(received_pm, HEADERS_SYNC_PATTERN, mode='valid') / len(HEADERS_SYNC_PATTERN)
    corr_data    = signal.correlate(received_pm, DATA_SYNC_PATTERN,    mode='valid') / len(DATA_SYNC_PATTERN)
    corr_end     = signal.correlate(received_pm, END_SYNC_PATTERN,     mode='valid') / len(END_SYNC_PATTERN)
    print(f"[DEC] 3×correlate: {time.time()-t}s")
    
    if np.max(corr_headers) < corr_threshold or np.max(corr_data) < corr_threshold or np.max(corr_end) < corr_threshold:
        raise ValueError(f"Sync not detected: headers={np.max(corr_headers)}, data={np.max(corr_data)}, end={np.max(corr_end)}")
    
    headers_start = np.argmax(corr_headers) + len(HEADERS_SYNC_PATTERN)
    data_start = np.argmax(corr_data) + len(DATA_SYNC_PATTERN)
    data_end = np.argmax(corr_end)
    
    if not (headers_start < data_start < data_end):
        raise ValueError(f"Invalid sync pattern order: headers_start={headers_start}, data_start={data_start}, data_end={data_end}")
    
    expected_data_bits = (data_end - data_start) * 8  # Approximate based on pixel range
    
    protected_headers = received_pm[headers_start:data_start - len(DATA_SYNC_PATTERN)]

    t3 = time.time()
    if USE_PRBS_FOR_HEADERS:
        # Despread headers
        n_groups_headers = len(protected_headers) // CHIP_LENGTH_FOR_HEADERS
        chips_headers = protected_headers[:n_groups_headers * CHIP_LENGTH_FOR_HEADERS].reshape(-1, CHIP_LENGTH_FOR_HEADERS)
        rx_bits_pm_headers = np.dot(chips_headers, HEADERS_PRBS) / CHIP_LENGTH_FOR_HEADERS
        rx_bits_headers = ((np.sign(rx_bits_pm_headers) + 1) / 2).astype(np.uint8)
    else:
        n_groups = len(protected_headers) // 3
        chips = protected_headers[:n_groups * 3].reshape(-1, 3)
        patterns = np.array([[-1, 1, -1], [1, -1, 1]], dtype=np.int32)
        corr = np.dot(chips, patterns.T) / 3
        rx_bits_headers = (np.argmax(corr, axis=1)).astype(np.uint8)
    print("[DEC] PRBS/map headers took: " + str(time.time() - t3))
    
    t4 = time.time()
    rx_bytes = np.packbits(rx_bits_headers).tobytes()
    t5 = time.time()
    print("[DEC] packbits headers took: " + str(t5 - t4))
    
    if USE_RS_FOR_HEADERS:
        try:
            t_rs_headers = time.time()
            decoded_headers = bytes(rsc.decode(bytearray(rx_bytes))[0])
            end_t_rs_headers = time.time()
            print("[DEC] rs decode for headers time " + str(end_t_rs_headers - t_rs_headers))
        except ReedSolomonError as e:
            raise ValueError(f"Header RS decoding failed: {e}")
    else:
        decoded_headers = rx_bytes
    t6 = time.time()
    
    sos_index = decoded_headers.find(b'\xff\xda')
    if not (decoded_headers.startswith(b'\xff\xd8') and sos_index != -1):
        raise ValueError(f"Invalid JPEG headers: start={decoded_headers[:2].hex()}, sos_index={sos_index}")
    
    # Despread data
    protected_data = received_pm[data_start:data_end]
    if USE_MARKER_CODEWORDS:
        t = time.time()
        data_bytes = _decode_data_with_codewords_popcnt(protected_data.astype(np.int8, copy=False), _TOKENS, _CODES_PACKED, MARKER_CODEWORD_LEN, MARKER_DET_THRESH)
        print("[DEC] Marker codewords decode took: " + str(time.time() - t))
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
        print("[DEC] Despread/map+packbits data took: " + str(time.time() - t))

    
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
        print("[DEC] RS data (chunked) took " + str(time.time() - t_rs_data))
    else:
        decoded_data = data_bytes
    
    t = time.time()
    fixed_data = fix_false_markers(decoded_data)
    print("[DEC] fix_false_markers took: " + str(time.time() - t))

    t7 = time.time()
    result = merge_jpeg(decoded_headers, fixed_data)
    print("[DEC] merge_jpeg took: " + str(time.time() - t7))
    print(f"Total decode time: {time.time() - t0} sec")
    return result

if __name__ == "__main__":
    cap = cv2.VideoCapture(PATH_TO_VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    cv2.namedWindow('Monitor', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Monitor', 2*FRAME_WIDTH + 20, 2*FRAME_HEIGHT + 20)

    # UI: noise slider (σ in [0..100])
    cv2.createTrackbar('Noise', 'Monitor', int(GAUSS_NOISE), 100, lambda v: None)
    cv2.setTrackbarPos('Noise', 'Monitor', int(GAUSS_NOISE))

    frame_count = 0
    while cap.isOpened():
        frame_count += 1
        frame_start = time.time()
        success, frame = cap.read()
        if not success:
            break
        h, w = frame.shape[:2]
        print(f"Original: {w}×{h}")
        
        frame_proc = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        encode_param = [(cv2.IMWRITE_JPEG_QUALITY), 30, cv2.IMWRITE_JPEG_RST_INTERVAL, 10]
        _, encoded_image = cv2.imencode(".jpg", frame_proc, encode_param)
        headers, compressed = split_jpeg(encoded_image.tobytes())
        
        # encode
        frame, tx_meta = encode_udp_to_frame(headers, compressed)
        
        # save the encoded frame
        # cv2.imwrite(f"encoded_{frame_count}.png", frame)
        # print(f"Encoded frame {frame_count} saved.")

        # read from slider
        sigma = float(cv2.getTrackbarPos('Noise', 'Monitor'))
        
        # add noise
        noisy = frame.astype(np.float32) + np.random.normal(0.0, sigma, frame.shape).astype(np.float32)
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)

        # analog video (for the GUI)
        analog_src = frame_proc
        analog_noisy = analog_src.astype(np.float32) + np.random.normal(0.0, sigma, analog_src.shape).astype(np.float32)
        analog_noisy = np.clip(analog_noisy, 0, 255).astype(np.uint8)

        # --- Pre-compute chip-level BER per section (independent of decode success) ---
        rx_pm = (2 * (noisy.ravel() > 127).astype(np.int8) - 1)
        tx_pm = tx_meta["stream_pm"]; idx = tx_meta["idx"]; L_end = idx["sync_e"][1]
        rx_pm = rx_pm[:L_end]

        s,e = idx["hdr"];    err_h  = int(np.count_nonzero(tx_pm[s:e] != rx_pm[s:e]));  tot_h  = e - s; ber_h  = (err_h/tot_h) if tot_h else 0.0
        s,e = idx["data"];   err_d  = int(np.count_nonzero(tx_pm[s:e] != rx_pm[s:e]));  tot_d  = e - s; ber_d  = (err_d/tot_d) if tot_d else 0.0
        s,e = idx["sync_h"]; err_sh = int(np.count_nonzero(tx_pm[s:e] != rx_pm[s:e]));  tot_sh = e - s
        s,e = idx["sync_d"]; err_sd = int(np.count_nonzero(tx_pm[s:e] != rx_pm[s:e]));  tot_sd = e - s
        s,e = idx["sync_e"]; err_se = int(np.count_nonzero(tx_pm[s:e] != rx_pm[s:e]));  tot_se = e - s

        err_sync = err_sh + err_sd + err_se
        tot_sync = tot_sh + tot_sd + tot_se
        ber_sync = (err_sync / tot_sync) if tot_sync else 0.0

        err_total  = err_h + err_d + err_sync
        bits_total = tot_h + tot_d + tot_sync
        ber_total  = (err_total / bits_total) if bits_total else 0.0

        line1 = f"BER stream: {100.0*ber_total:.2f}%  ({err_total}/{bits_total} chips)"
        line2 = f"H: {100.0*ber_h:.2f}%  D: {100.0*ber_d:.2f}%  Sync: {100.0*ber_sync:.2f}%"

        try:
            # decode
            decoded_data = decode_frame_to_udp(noisy)
            decoded_np = np.frombuffer(decoded_data, dtype=np.uint8)
            decoded_img = cv2.imdecode(decoded_np, cv2.IMREAD_COLOR)
            if decoded_img is None:
                # show black recovered frame
                frame_to_show = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
                annotated = frame_to_show.copy(); y0 = 22
                x1 = FRAME_WIDTH - 10 - cv2.getTextSize(line1, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0][0]
                cv2.putText(annotated, line1, (x1, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 3, cv2.LINE_AA)
                cv2.putText(annotated, line1, (x1, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)
                y0 += 20
                x2 = FRAME_WIDTH - 10 - cv2.getTextSize(line2, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0][0]
                cv2.putText(annotated, line2, (x2, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 3, cv2.LINE_AA)
                cv2.putText(annotated, line2, (x2, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)
                frame_to_show = annotated

            else:
                if decoded_img.shape[1] != FRAME_WIDTH or decoded_img.shape[0] != FRAME_HEIGHT:
                    decoded_img = cv2.resize(decoded_img, (FRAME_WIDTH, FRAME_HEIGHT))
                frame_to_show = decoded_img

                annotated = frame_to_show.copy(); y0 = 22
                x1 = FRAME_WIDTH - 10 - cv2.getTextSize(line1, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0][0]
                cv2.putText(annotated, line1, (x1, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 3, cv2.LINE_AA)
                cv2.putText(annotated, line1, (x1, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)
                y0 += 20
                x2 = FRAME_WIDTH - 10 - cv2.getTextSize(line2, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0][0]
                cv2.putText(annotated, line2, (x2, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 3, cv2.LINE_AA)
                cv2.putText(annotated, line2, (x2, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)
                frame_to_show = annotated

        except ValueError as e:
            print(f"Error: {e}")
            # show black recovered frame inside the single-Window mosaic
            frame_to_show = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
            annotated = frame_to_show.copy(); y0 = 22
            x1 = FRAME_WIDTH - 10 - cv2.getTextSize(line1, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0][0]
            cv2.putText(annotated, line1, (x1, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 3, cv2.LINE_AA)
            cv2.putText(annotated, line1, (x1, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)
            y0 += 20
            x2 = FRAME_WIDTH - 10 - cv2.getTextSize(line2, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0][0]
            cv2.putText(annotated, line2, (x2, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 3, cv2.LINE_AA)
            cv2.putText(annotated, line2, (x2, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)
            frame_to_show = annotated

        orig_vis = _label(_to_bgr(frame_proc), 'Original')
        analog_vis = _label(_to_bgr(analog_noisy), 'Analog')
        enc_vis   = _label(_to_bgr(noisy), 'Encoded+Noise')
        rec_vis   = _label(_to_bgr(frame_to_show), 'Recovered')

        mosaic = _compose_grid(orig_vis, analog_vis, enc_vis, rec_vis, gap=20)
        cv2.imshow('Monitor', mosaic)
        delay_ms = max(1, int(1000.0 / fps - (time.time() - frame_start) * 1000.0))
        if cv2.waitKey(delay_ms) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

