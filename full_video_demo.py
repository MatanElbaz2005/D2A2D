import numpy as np
try:
    from creedsolo import RSCodec, ReedSolomonError
except ImportError:
    print("cant find creedsolo, using reedsolo instead")
    from reedsolo import RSCodec, ReedSolomonError
import cv2
import time
from protected_jpeg import split_jpeg, merge_jpeg, fix_false_markers
from helpers_files.helpers import generate_prbs, _mseq_127_taps_7_1, _mseq_127_taps_7_3, gold127, _decode_data_with_codewords_popcnt
from helpers_files.helpers import _build_marker_codewords_gold, _is_marker_token_at, _encode_data_with_codewords_fast, _decode_data_with_codewords_fast
from helpers_files.gui_helpers import _to_bgr, _compose_grid, _label
from helpers_files.runtime_helpers import _rt_init, _rt_set_frame, _rt_record, _rt_print, _rt_flush_if_ready
from helpers_files.camera_helpers import open_capture
from helpers_files.helpers import _encode_len_block_chips, _decode_len_block_chips, decode_codewords, _encode_len_block_chips_dataonly, _decode_len_block_chips_dataonly
from key_files.encode import encode_udp_to_frame_dataonly
import binxcorr
import yaml
from helpers_files.config_helpers import _cfg, get_rsc, get_marker_codebook, get_headers_sync, get_prbs
from key_files.decode import decode_frame_to_udp

cfg = _cfg()

FRAME_WIDTH  = cfg["frame"]["width"]
FRAME_HEIGHT = cfg["frame"]["height"]

USE_RS_FOR_HEADERS = cfg["rs"]["use_for_headers"]
USE_RS_FOR_DATA    = cfg["rs"]["use_for_data"]
ECC_SYMBOLS        = cfg["rs"]["ecc_symbols"]
CHUNK_BYTES        = cfg["rs"]["chunk_bytes"]

USE_PRBS_FOR_HEADERS   = cfg["prbs"]["use_for_headers"]
USE_PRBS_FOR_DATA      = cfg["prbs"]["use_for_data"]
CHIP_LENGTH_FOR_HEADERS = cfg["prbs"]["chip_length_for_headers"]
CHIP_LENGTH_FOR_DATA    = cfg["prbs"]["chip_length_for_data"]
DATA_PRBS_POLY          = cfg["prbs"]["data_prbs_poly"]

LENGTH_BITS_PER_FIELD = cfg["length"]["bits_per_field"]
LENGTH_CHIP_LENGTH    = cfg["length"]["chip_length"]

USE_MARKER_CODEWORDS = cfg["markers"]["use"]
MARKER_CODEWORD_LEN  = cfg["markers"]["codeword_len"]
MARKER_DET_THRESH    = cfg["markers"]["det_thresh"]

INPUT_SOURCE   = cfg["camera"]["input_source"]
CAMERA_INDEX   = cfg["camera"]["camera_index"]
TARGET_FPS     = cfg["camera"]["target_fps"]
PATH_TO_VIDEO  = cfg["camera"]["path_to_video"]

GAUSS_NOISE = cfg["noise"]["gauss_noise"]

save_runtime       = cfg["runtime"]["save"]
OS                 = cfg["runtime"]["os"]
RUNTIME_START_FRAME = cfg["runtime"]["start_frame"]
RUNTIME_NUM_FRAMES  = cfg["runtime"]["num_frames"]

_RUNTIME = {
    "enabled": False,
    "frame": 0,
    "collected": {},
    "flushed": False,
    "filename": None
}

HEADER_TEMPLATE: bytes | None = None
HEADER_TEMPLATE_READY: bool = False

if USE_MARKER_CODEWORDS:
    codewords_time = time.time()
    _TOKENS, _CODES = get_marker_codebook()
    _CODES_PACKED = np.packbits((_CODES > 0).astype(np.uint8), axis=1)

    norm = (_CODES @ _CODES.T) / _CODES.shape[1]
    for i in range(norm.shape[0]):
        norm[i, i] = 0.0

    print(f"[codewords] L={MARKER_CODEWORD_LEN} (token -> codeword 01)")
    for tok, cw in zip(_TOKENS, _CODES):
        s01 = ''.join('1' if int(v) > 0 else '0' for v in cw.tolist())
    _rt_print(_RUNTIME, "[ENC] Build marker codewords took: ", time.time() - codewords_time)

perp_rsc_time = time.time()
rsc = get_rsc()
_rt_print(_RUNTIME, "[ENC] preper RS took ", time.time() - perp_rsc_time)

# Sync patterns (gold codes, ±1)
HEADERS_SYNC_PATTERN = get_headers_sync()

_HDR_PACK  = np.packbits((HEADERS_SYNC_PATTERN > 0).astype(np.uint8), bitorder="big")

def _last_byte_mask(bit_length: int) -> np.uint8:
    rem = bit_length % 8
    if rem == 0:
        return np.uint8(0xFF)
    return np.uint8((0xFF << (8 - rem)) & 0xFF)

_HDR_MASK = _last_byte_mask(HEADERS_SYNC_PATTERN.size)

# PRBS for spreading (if enabled)
prbs_headers_time = time.time()
HEADERS_PRBS = get_prbs(CHIP_LENGTH_FOR_HEADERS, tuple(DATA_PRBS_POLY), seed=3) if USE_PRBS_FOR_HEADERS else None
if USE_PRBS_FOR_HEADERS: _rt_print(_RUNTIME, "[ENC] generate PRBS headers took: ", time.time() - prbs_headers_time)

# PRBS for the length block (longer than headers for extra gain)
prbs_length_time = time.time()
LENGTH_PRBS  = get_prbs(LENGTH_CHIP_LENGTH,     tuple(DATA_PRBS_POLY), seed=3) if USE_PRBS_FOR_HEADERS else None
if USE_PRBS_FOR_HEADERS: _rt_print(_RUNTIME, "[ENC] generate PRBS length took: ", time.time() - prbs_length_time)

prbs_data_time = time.time()
DATA_PRBS    = get_prbs(CHIP_LENGTH_FOR_DATA,   tuple(DATA_PRBS_POLY), seed=3) if USE_PRBS_FOR_DATA    else None
if USE_PRBS_FOR_DATA: _rt_print(_RUNTIME, "[ENC] generate PRBS data took: ", time.time() - prbs_data_time)


if __name__ == "__main__":
    cv2.setUseOptimized(True)
    cv2.setNumThreads(0)
    _rt_init(save_runtime, _RUNTIME, OS, sample_start=RUNTIME_START_FRAME, sample_count=RUNTIME_NUM_FRAMES)
    cap, fps = open_capture()

    cv2.namedWindow('Monitor', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Monitor', 2*FRAME_WIDTH + 20, 2*FRAME_HEIGHT + 20)

    # UI: noise slider (σ in [0..100])
    cv2.createTrackbar('Noise', 'Monitor', int(GAUSS_NOISE), 100, lambda v: None)
    cv2.setTrackbarPos('Noise', 'Monitor', int(GAUSS_NOISE))

    frame_count = 0
    while cap.isOpened():
        frame_count += 1
        _rt_set_frame(frame_count, _RUNTIME)
        if RUNTIME_START_FRAME <= frame_count < RUNTIME_START_FRAME + RUNTIME_NUM_FRAMES:
            _RUNTIME["enabled"] = True
        else:
            _RUNTIME["enabled"] = False
        frame_start = time.time()
        t = time.time()
        success, frame = cap.read()
        _rt_print(_RUNTIME, "[LOOP] cap.read: ", time.time()-t, " s")
        if not success:
            break
        h, w = frame.shape[:2]
        print(f"Original: {w}×{h}")
        
        t = time.time()
        frame_proc = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT), interpolation=cv2.INTER_NEAREST)
        _rt_print(_RUNTIME, "[LOOP] resize: ", time.time()-t, " s")
        encode_param = [(cv2.IMWRITE_JPEG_QUALITY), 70, cv2.IMWRITE_JPEG_RST_INTERVAL, 10]
        t = time.time()
        _, encoded_image = cv2.imencode(".jpg", frame_proc, encode_param)
        _rt_print(_RUNTIME, "[LOOP] imencode: ", time.time()-t, " s")
        t = time.time()
        headers, compressed = split_jpeg(encoded_image.tobytes())
        _rt_print(_RUNTIME, "[LOOP] split_jpeg: ", time.time()-t, " s")
        if not HEADER_TEMPLATE_READY:
            HEADER_TEMPLATE = headers
            HEADER_TEMPLATE_READY = True
            continue
        
        # encode
        t = time.time()
        frame, tx_meta = encode_udp_to_frame_dataonly(compressed, _RUNTIME=_RUNTIME)
        _rt_print(_RUNTIME, "[ENC] encode_udp_to_frame (outer): ", time.time()-t, " s")
        
        # save the encoded frame
        # cv2.imwrite(f"encoded_{frame_count}.png", frame)
        # print(f"Encoded frame {frame_count} saved.")

        # read from slider
        t = time.time()
        sigma = float(cv2.getTrackbarPos('Noise', 'Monitor'))
        _rt_print(_RUNTIME, "[GUI] read slider: ", time.time()-t, " s")
        
        # add noise
        t = time.time()
        noisy = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR).astype(np.float32)
        noisy += np.random.normal(0.0, sigma, noisy.shape).astype(np.float32)
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)
        _rt_print(_RUNTIME, "[LOOP] add noise (chips): ", time.time()-t, " s")

        # analog video (for the GUI)
        t = time.time()
        analog_src = frame_proc
        analog_noisy = analog_src.astype(np.float32) + np.random.normal(0.0, sigma, analog_src.shape).astype(np.float32)
        analog_noisy = np.clip(analog_noisy, 0, 255).astype(np.uint8)
        _rt_print(_RUNTIME, "[GUI] add noise (analog): ", time.time()-t, " s")

        t = time.time()
        # --- Pre-compute chip-level BER per section ---
        noisy_gray = cv2.cvtColor(noisy, cv2.COLOR_BGR2GRAY)
        rx_pm = (2 * (noisy_gray.ravel() > 127).astype(np.int8) - 1)
        tx_pm = tx_meta["stream_pm"]; idx = tx_meta["idx"]; L_end = idx["data"][1]
        rx_pm = rx_pm[:L_end]

        s,e = idx["data"];   err_d  = int(np.count_nonzero(tx_pm[s:e] != rx_pm[s:e]));  tot_d  = e - s; ber_d  = (err_d/tot_d) if tot_d else 0.0
        s,e = idx["sync"];  err_sh = int(np.count_nonzero(tx_pm[s:e] != rx_pm[s:e]));  tot_sh = e - s
        s,e = idx["len"];   err_sl = int(np.count_nonzero(tx_pm[s:e] != rx_pm[s:e]));  tot_sl = e - s

        err_sync = err_sh + err_sl
        tot_sync = tot_sh + tot_sl
        ber_sync = (err_sync / tot_sync) if tot_sync else 0.0

        err_total  = err_d + err_sync
        bits_total = tot_d + tot_sync
        ber_total  = (err_total / bits_total) if bits_total else 0.0

        line2 = f"D: {100.0*ber_d:.2f}%  Sync: {100.0*ber_sync:.2f}%"

        try:
            # decode
            t = time.time()
            decoded_data = decode_frame_to_udp(noisy_gray, _RUNTIME, HEADER_TEMPLATE_READY, HEADER_TEMPLATE)
            _rt_print(_RUNTIME, "[DEC] decode_frame_to_udp (outer): ", time.time()-t, " s")
            t = time.time()
            decoded_np = np.frombuffer(decoded_data, dtype=np.uint8)
            decoded_img = cv2.imdecode(decoded_np, cv2.IMREAD_COLOR)
            _rt_print(_RUNTIME, "[GUI] imdecode recovered: ", time.time()-t, " s")
            if decoded_img is None:
                # show black recovered frame
                frame_to_show = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
                annotated = frame_to_show.copy(); y0 = 22
                x1 = FRAME_WIDTH - 10 - cv2.getTextSize(line2, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0][0]
                cv2.putText(annotated, line2, (x1, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 3, cv2.LINE_AA)
                cv2.putText(annotated, line2, (x1, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)
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
                x1 = FRAME_WIDTH - 10 - cv2.getTextSize(line2, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0][0]
                cv2.putText(annotated, line2, (x1, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 3, cv2.LINE_AA)
                cv2.putText(annotated, line2, (x1, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)
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
            x1 = FRAME_WIDTH - 10 - cv2.getTextSize(line2, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0][0]
            cv2.putText(annotated, line2, (x1, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 3, cv2.LINE_AA)
            cv2.putText(annotated, line2, (x1, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)
            y0 += 20
            x2 = FRAME_WIDTH - 10 - cv2.getTextSize(line2, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0][0]
            cv2.putText(annotated, line2, (x2, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 3, cv2.LINE_AA)
            cv2.putText(annotated, line2, (x2, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)
            frame_to_show = annotated

        t = time.time()
        orig_vis = _label(_to_bgr(frame_proc), 'Original')
        analog_vis = _label(_to_bgr(analog_noisy), 'Analog')
        enc_vis   = _label(_to_bgr(noisy), 'Encoded+Noise')
        rec_vis   = _label(_to_bgr(frame_to_show), 'Recovered')
        _rt_print(_RUNTIME, "[GUI] build labels: ", time.time()-t, " s")

        t = time.time()
        mosaic = _compose_grid(orig_vis, analog_vis, enc_vis, rec_vis, gap=20)
        _rt_print(_RUNTIME, "[GUI] compose mosaic: ", time.time()-t, " s")

        t = time.time()
        cv2.imshow('Monitor', mosaic)
        _rt_print(_RUNTIME, "[GUI] imshow: ", time.time()-t, " s")
        _rt_print(_RUNTIME, "[LOOP] frame total: ", time.time()-frame_start, " s")
        _rt_flush_if_ready(_RUNTIME, OS)
        delay_ms = max(1, int(1000.0 / fps - (time.time() - frame_start) * 1000.0))
        if cv2.waitKey(delay_ms) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

