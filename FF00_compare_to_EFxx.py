# FF00_compare_to_EFxx.py
import json, csv, time
from pathlib import Path

import numpy as np
import cv2
import binxcorr

import full_video_demo as core

from protected_jpeg import split_jpeg
from helpers_files.camera_helpers import open_capture
from helpers_files.gui_helpers import _to_bgr, _label
from helpers_files.debug_helpers import (
    rebuild_two_variants, ssim_color, accumulate_stats, summarize_metrics, hstack3,
    make_info_bar, build_summary_json
)

FRAME_WIDTH       = core.FRAME_WIDTH
FRAME_HEIGHT      = core.FRAME_HEIGHT
TARGET_FPS        = core.TARGET_FPS
INPUT_SOURCE      = core.INPUT_SOURCE
CAMERA_INDEX      = core.CAMERA_INDEX
PATH_TO_VIDEO     = core.PATH_TO_VIDEO
USE_MARKER_CODEWORDS = core.USE_MARKER_CODEWORDS

encode_udp_to_frame = core.encode_udp_to_frame

MAX_FRAMES   = 100
JPEG_QUALITY = 70
JPEG_RST_INTERVAL = 10
WINDOW_TITLE = "Fix-Strategy Eval (Original | FF00 | EFxx)"

OUT_CSV  = Path("fix_eval_results.csv")
OUT_JSON = Path("fix_eval_summary.json")


def _resize_if_needed(img: np.ndarray) -> np.ndarray:
    if img.shape[1] != FRAME_WIDTH or img.shape[0] != FRAME_HEIGHT:
        return cv2.resize(img, (FRAME_WIDTH, FRAME_HEIGHT))
    return img


def decode_frame_to_raw(frame: np.ndarray, corr_threshold: float = 0.9):
    if frame.shape != (FRAME_HEIGHT, FRAME_WIDTH):
        raise ValueError(f"Frame size mismatch: expected {FRAME_HEIGHT}x{FRAME_WIDTH}")

    received_pm = (2 * (frame.ravel() > 127).astype(np.int8) - 1)

    search_end = max(len(core.HEADERS_SYNC_PATTERN), int(received_pm.size * 0.10))
    h_corr, h_idx = binxcorr.correlate_sliding_bin_argmax(
        received_pm, core.HEADERS_SYNC_PATTERN, 0, search_end, False
    )
    if h_corr < corr_threshold:
        raise ValueError(f"Sync not detected (headers): {h_corr:.3f}")

    sync_start = int(h_idx)
    sync_end   = sync_start + len(core.HEADERS_SYNC_PATTERN)

    len_bits_total = 2 * core.LENGTH_BITS_PER_FIELD
    chips_per_bit  = (core.LENGTH_CHIP_LENGTH if core.USE_PRBS_FOR_HEADERS else 3)
    len_block_chips = len_bits_total * chips_per_bit

    len_block_rx = received_pm[sync_end: sync_end + len_block_chips]
    hdr_chips_len, data_chips_len = core._decode_len_block_chips(
        len_block_rx,
        core.USE_PRBS_FOR_HEADERS,
        core.LENGTH_CHIP_LENGTH,
        (core.LENGTH_PRBS if core.USE_PRBS_FOR_HEADERS else None),
        core.LENGTH_BITS_PER_FIELD
    )

    headers_start = sync_end + len_block_chips
    headers_end   = headers_start + hdr_chips_len
    data_start    = headers_end
    data_end      = data_start + data_chips_len

    if not (headers_start < data_start < data_end):
        raise ValueError(f"Invalid sync pattern order: headers_start={headers_start}, data_start={data_start}, data_end={data_end}")

    protected_headers = received_pm[headers_start:headers_end]
    if core.USE_PRBS_FOR_HEADERS:
        n_groups_headers = len(protected_headers) // core.CHIP_LENGTH_FOR_HEADERS
        chips_headers = protected_headers[:n_groups_headers * core.CHIP_LENGTH_FOR_HEADERS].reshape(-1, core.CHIP_LENGTH_FOR_HEADERS)
        rx_bits_pm_headers = np.dot(chips_headers, core.HEADERS_PRBS) / core.CHIP_LENGTH_FOR_HEADERS
        rx_bits_headers = ((np.sign(rx_bits_pm_headers) + 1) / 2).astype(np.uint8)
    else:
        n_groups = len(protected_headers) // 3
        chips = protected_headers[:n_groups * 3].reshape(-1, 3)
        patterns = np.array([[-1, 1, -1], [1, -1, 1]], dtype=np.int32)
        corr = np.dot(chips, patterns.T) / 3
        rx_bits_headers = (np.argmax(corr, axis=1)).astype(np.uint8)

    rx_bytes = np.packbits(rx_bits_headers).tobytes()

    if core.USE_RS_FOR_HEADERS:
        try:
            decoded_headers = bytes(core.rsc.decode(bytearray(rx_bytes))[0])
        except Exception as e:
            raise ValueError(f"Header RS decoding failed: {e}")
    else:
        decoded_headers = rx_bytes

    sos_index = decoded_headers.find(b'\xff\xda')
    if not (decoded_headers.startswith(b'\xff\xd8') and sos_index != -1):
        raise ValueError(f"Invalid JPEG headers: start={decoded_headers[:2].hex()}, sos_index={sos_index}")

    protected_data = received_pm[data_start:data_end]
    token_starts = None

    if core.USE_MARKER_CODEWORDS:
        data_dec = core._decode_data_with_codewords_popcnt(
            protected_data.astype(np.int8, copy=False),
            core._TOKENS, core._CODES_PACKED,
            core.MARKER_CODEWORD_LEN, core.MARKER_DET_THRESH,
            return_token_positions=True
        )
        if isinstance(data_dec, tuple):
            data_bytes, token_starts = data_dec
        else:
            data_bytes = data_dec
            token_starts = None
    else:
        if core.USE_PRBS_FOR_DATA:
            n_groups_data = len(protected_data) // core.CHIP_LENGTH_FOR_DATA
            chips_data = protected_data[:n_groups_data * core.CHIP_LENGTH_FOR_DATA].reshape(-1, core.CHIP_LENGTH_FOR_DATA)
            rx_bits_pm_data = np.dot(chips_data, core.DATA_PRBS) / core.CHIP_LENGTH_FOR_DATA
            rx_bits_data = ((np.sign(rx_bits_pm_data) + 1) / 2).astype(np.uint8)
        else:
            rx_bits_data = ((protected_data + 1) / 2).astype(np.uint8)
        data_bytes = np.packbits(rx_bits_data).tobytes()

    if core.USE_RS_FOR_DATA:
        decoded_chunks = []
        i = 0
        N = core.CHUNK_BYTES + core.ECC_SYMBOLS
        while i + N <= len(data_bytes):
            blk = data_bytes[i:i+N]
            try:
                decoded_chunks.append(bytes(core.rsc.decode(bytearray(blk))[0]))
            except Exception:
                decoded_chunks.append(blk[:core.CHUNK_BYTES])
            i += N
        if i < len(data_bytes):
            blk = data_bytes[i:]
            try:
                decoded_chunks.append(bytes(core.rsc.decode(bytearray(blk))[0]))
            except Exception:
                k_last = max(0, len(blk) - core.ECC_SYMBOLS)
                decoded_chunks.append(blk[:k_last])
        decoded_data = b"".join(decoded_chunks)
    else:
        decoded_data = data_bytes

    return decoded_headers, decoded_data, (token_starts if core.USE_MARKER_CODEWORDS else None)


def main():
    cap, fps = open_capture(core.INPUT_SOURCE, "raspberry_pi", core.PATH_TO_VIDEO,
                            core.CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT, TARGET_FPS)

    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_TITLE, 3*FRAME_WIDTH + 40, FRAME_HEIGHT + 60)

    GAUSS_NOISE = 50
    cv2.createTrackbar('Noise', WINDOW_TITLE, int(GAUSS_NOISE), 100, lambda v: None)
    cv2.setTrackbarPos('Noise', WINDOW_TITLE, int(GAUSS_NOISE))

    records = []
    totals  = {}

    frame_idx = 0
    t0 = time.time()

    while cap.isOpened() and frame_idx < MAX_FRAMES:
        frame_idx += 1
        ret, frame = cap.read()
        if not ret:
            break

        frame_proc = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

        encode_param = [(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY, cv2.IMWRITE_JPEG_RST_INTERVAL, JPEG_RST_INTERVAL]
        ok, encoded_image = cv2.imencode(".jpg", frame_proc, encode_param)
        if not ok:
            print("[ERR] imencode failed")
            continue

        ref_img = cv2.imdecode(encoded_image, cv2.IMREAD_COLOR)
        if ref_img is None:
            print("[ERR] ref_img decode failed")
            continue
        ref_img = _resize_if_needed(ref_img)

        headers, compressed = split_jpeg(encoded_image.tobytes())

        chips_frame, _ = encode_udp_to_frame(headers, compressed)

        sigma = float(cv2.getTrackbarPos('Noise', WINDOW_TITLE))
        noisy = chips_frame.astype(np.float32) + np.random.normal(0.0, sigma, chips_frame.shape).astype(np.float32)
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)

        decoded_headers, data_bytes, token_starts = decode_frame_to_raw(noisy)

        (jpg_ff00, stats_ff), (jpg_efxx, stats_ef) = rebuild_two_variants(
            decoded_headers, data_bytes, USE_MARKER_CODEWORDS, token_starts
        )

        img_ff = cv2.imdecode(np.frombuffer(jpg_ff00, np.uint8), cv2.IMREAD_COLOR)
        img_ef = cv2.imdecode(np.frombuffer(jpg_efxx, np.uint8), cv2.IMREAD_COLOR)
        if img_ff is None or img_ef is None:
            print("[ERR] variant decode failed")
            continue

        img_ff = _resize_if_needed(img_ff)
        img_ef = _resize_if_needed(img_ef)

        psnr_ff = cv2.PSNR(ref_img, img_ff)
        psnr_ef = cv2.PSNR(ref_img, img_ef)
        ssim_ff = ssim_color(ref_img, img_ff)
        ssim_ef = ssim_color(ref_img, img_ef)

        print(f"[EVAL f={frame_idx}] FF00: PSNR={psnr_ff:.2f} SSIM={ssim_ff:.4f} | EFxx: PSNR={psnr_ef:.2f} SSIM={ssim_ef:.4f}")

        records.append({
            "frame": frame_idx,
            "sigma": sigma,
            "psnr_ff": psnr_ff, "ssim_ff": ssim_ff,
            "psnr_ef": psnr_ef, "ssim_ef": ssim_ef,
            "ff_false_rst": stats_ff.get("false_rst", 0),
            "ff_illegal_ffxx": stats_ff.get("illegal_ffxx", 0),
            "ff_to_ff00": stats_ff.get("to_ff00", 0),
            "ff_to_efxx": stats_ff.get("to_efxx", 0),
            "ef_false_rst": stats_ef.get("false_rst", 0),
            "ef_illegal_ffxx": stats_ef.get("illegal_ffxx", 0),
            "ef_to_ff00": stats_ef.get("to_ff00", 0),
            "ef_to_efxx": stats_ef.get("to_efxx", 0),
        })
        accumulate_stats(totals, stats_ff, "ff_")
        accumulate_stats(totals, stats_ef, "ef_")

        vis_orig = _label(_to_bgr(frame_proc), "Original")
        vis_ff   = _label(_to_bgr(img_ff),     "FF00")
        vis_ef   = _label(_to_bgr(img_ef),     "EFxx")
        mosaic   = hstack3(vis_orig, vis_ff, vis_ef)

        info_txt = (
            f"frame={frame_idx}  σ={sigma:.0f}  |  "
            f"FF00: PSNR {psnr_ff:.2f} SSIM {ssim_ff:.4f}   "
            f"EFxx: PSNR {psnr_ef:.2f} SSIM {ssim_ef:.4f}"
        )
        info_bar = make_info_bar(mosaic.shape[1], info_txt, height=36)
        stacked  = cv2.vconcat([mosaic, info_bar])
        cv2.imshow(WINDOW_TITLE, stacked)

        delay_ms = max(1, int(1000.0 / (fps or TARGET_FPS)))
        if cv2.waitKey(delay_ms) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if records:
        with OUT_CSV.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
            w.writeheader()
            for r in records:
                w.writerow(r)

    elapsed = time.time() - t0
    summary = build_summary_json(records, totals, frames=len(records), elapsed_sec=elapsed)
    with OUT_JSON.open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"[DONE] Wrote CSV:  {OUT_CSV.resolve()}")
    print(f"[DONE] Wrote JSON: {OUT_JSON.resolve()}")


if __name__ == "__main__":
    main()
