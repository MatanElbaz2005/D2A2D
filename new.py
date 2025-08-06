import numpy as np
from reedsolo import RSCodec, ReedSolomonError
from scipy import signal
import cv2
import time
from protected_jpeg import jpg_parse, jpg_build, fix_false_markers
from helpers import generate_prbs

# Config
FRAME_WIDTH = 720
FRAME_HEIGHT = 480  # NTSC
ECC_SYMBOLS = 32
CHIP_LENGTH_FOR_HEADERS = 5
CHIP_LENGTH_FOR_DATA = 2
DATA_PRBS_POLY = [8, 2]
FORMAT_IMAGE = 'jpg'
MEMORY = [6]
G_MATRIX = [[0o133, 0o171]]
TB_LENGTH = 15
PATH_TO_VIDEO = r"/home/pi/Documents/matan/code/D2A2D/1572378-sd_960_540_24fps.mp4"
USE_PRBS = True
USE_RS = False
rsc = RSCodec(ECC_SYMBOLS)

# Sync patterns (Barker codes, ±1)
HEADERS_SYNC_PATTERN = np.array([1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1], dtype=np.int32)
DATA_SYNC_PATTERN = np.array([1, -1, 1, -1, 1, 1, -1, -1, 1, 1, 1, 1, 1], dtype=np.int32)
END_SYNC_PATTERN = np.array([-1, -1, -1, -1, -1, 1, 1, -1, -1, 1, -1, 1, -1, 1, 1, -1, 1, -1, -1, 1], dtype=np.int32)  # Extended pattern

# PRBS for spreading (if enabled)
HEADERS_PRBS = generate_prbs(CHIP_LENGTH_FOR_HEADERS, DATA_PRBS_POLY, 3) if USE_PRBS else None
DATA_PRBS = generate_prbs(CHIP_LENGTH_FOR_DATA, DATA_PRBS_POLY, 3) if USE_PRBS else None

def encode_udp_to_frame(headers: bytes, data: bytes) -> np.ndarray:
    start_time = time.time()
    
    print(f"[encode] Header size: {len(headers)} bytes")
    print(f"[encode] First 10 header bytes: {headers[:10].hex()}")
    print(f"[encode] Last 10 header bytes: {headers[-10:].hex()}")
    
    if USE_RS:
        coded_headers = rsc.encode(headers)
        print(f"[encode] RS-coded header size: {len(coded_headers)} bytes")
    else:
        coded_headers = headers
        print(f"[encode] No RS, header size: {len(coded_headers)} bytes")
    
    header_bits = np.unpackbits(np.frombuffer(coded_headers, dtype=np.uint8))
    print(f"[encode] Header bits length: {len(header_bits)}")
    header_bits_pm = 2.0 * header_bits - 1
    
    if USE_PRBS:
        repeated_bits = np.repeat(header_bits_pm, CHIP_LENGTH_FOR_HEADERS)
        tiled_prbs = np.tile(HEADERS_PRBS, len(header_bits))
        protected_headers = repeated_bits * tiled_prbs
        print(f"[encode] Using PRBS, protected headers length: {len(protected_headers)} bits")
    else:
        mapping = {0: np.array([-1, 1, -1]), 1: np.array([1, -1, 1])}
        protected_headers = np.concatenate([mapping[bit] for bit in header_bits])
        print(f"[encode] Using 3-bit mapping, protected headers length: {len(protected_headers)} bits")
    
    data_bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
    print(f"[encode] Data bits length: {len(data_bits)}")

    data_bits_pm = 2.0 * data_bits - 1
    if USE_PRBS:
        repeated_data_bits = np.repeat(data_bits_pm, CHIP_LENGTH_FOR_DATA)
        tiled_prbs = np.tile(DATA_PRBS, len(data_bits))
        protected_data = repeated_data_bits * tiled_prbs
        print(f"[encode] Using PRBS, protected data length: {len(protected_data)} bits")
    else:
        protected_data = data_bits_pm

    full_stream = np.concatenate((
        HEADERS_SYNC_PATTERN,
        protected_headers,
        DATA_SYNC_PATTERN,
        protected_data,
        END_SYNC_PATTERN
    ))
    print(f"[encode] Full stream length: {len(full_stream)} bits")
    
    total_pixels = FRAME_WIDTH * FRAME_HEIGHT
    if len(full_stream) > total_pixels:
        raise ValueError(f"Data too large: {len(full_stream)} bits > {total_pixels} pixels")
    
    full_stream = ((full_stream + 1) / 2 * 255).astype(np.uint8)
    full_stream = np.pad(full_stream, (0, total_pixels - len(full_stream)), 'constant')
    frame = full_stream.reshape((FRAME_HEIGHT, FRAME_WIDTH))
    
    print(f"[encode] Frame shape: {frame.shape}")
    print(f"Encode took: {time.time() - start_time} sec")
    return frame

def decode_frame_to_udp(frame: np.ndarray, corr_threshold: float = 0.8) -> bytes:
    t0 = time.time()
    if frame.shape != (FRAME_HEIGHT, FRAME_WIDTH):
        raise ValueError(f"Frame size mismatch: expected {FRAME_HEIGHT}x{FRAME_WIDTH}")
    
    received_pm = 2 * (frame.ravel() > 127).astype(np.int32) - 1
    t1 = time.time()
    print(f"[perf] threshold→±1: {t1 - t0} sec")
    
    corr_headers = signal.correlate(received_pm, HEADERS_SYNC_PATTERN, mode='valid') / len(HEADERS_SYNC_PATTERN)
    corr_data = signal.correlate(received_pm, DATA_SYNC_PATTERN, mode='valid') / len(DATA_SYNC_PATTERN)
    corr_end = signal.correlate(received_pm, END_SYNC_PATTERN, mode='valid') / len(END_SYNC_PATTERN)
    t2 = time.time()
    print(f"[perf] correlations: {t2 - t1} sec")
    print(f"[decode] Header sync max corr: {np.max(corr_headers)} at pos {np.argmax(corr_headers)}")
    print(f"[decode] Data sync max corr: {np.max(corr_data)} at pos {np.argmax(corr_data)}")
    print(f"[decode] End sync max corr: {np.max(corr_end)} at pos {np.argmax(corr_end)}")
    
    if np.max(corr_headers) < corr_threshold or np.max(corr_data) < corr_threshold or np.max(corr_end) < corr_threshold:
        raise ValueError(f"Sync not detected: headers={np.max(corr_headers)}, data={np.max(corr_data)}, end={np.max(corr_end)}")
    
    headers_start = np.argmax(corr_headers) + len(HEADERS_SYNC_PATTERN)
    data_start = np.argmax(corr_data) + len(DATA_SYNC_PATTERN)
    data_end = np.argmax(corr_end)
    
    if not (headers_start < data_start < data_end):
        raise ValueError(f"Invalid sync pattern order: headers_start={headers_start}, data_start={data_start}, data_end={data_end}")
    print(f"[decode] Headers range: {headers_start}:{data_start - len(DATA_SYNC_PATTERN)}")
    print(f"[decode] Data range: {data_start}:{data_end}")
    
    expected_data_bits = (data_end - data_start) * 8  # Approximate based on pixel range
    print(f"[decode] Expected data bits (approx): {expected_data_bits}")
    
    protected_headers = received_pm[headers_start:data_start - len(DATA_SYNC_PATTERN)]
    print(f"[decode] Protected headers length: {len(protected_headers)} bits")
    t3 = time.time()
    print(f"[perf] extraction: {t3 - t2} sec")
    
    if USE_PRBS:
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
    print(f"[decode] Despread bits length: {len(rx_bits_headers)}")
    print(f"[decode] First 16 despread bits: {rx_bits_headers[:16].tolist()}")
    print(f"[decode] Last 16 despread bits: {rx_bits_headers[-16:].tolist()}")
    
    t4 = time.time()
    print(f"[perf] despreading: {t4 - t3} sec")
    
    rx_bytes = np.packbits(rx_bits_headers).tobytes()
    print(f"[decode] Packed bytes length: {len(rx_bytes)}")
    print(f"[decode] First 10 packed bytes: {rx_bytes[:10].hex()}")
    print(f"[decode] Last 10 packed bytes: {rx_bytes[-10:].hex()}")
    t5 = time.time()
    print(f"[perf] packbits: {t5 - t4} sec")
    
    if USE_RS:
        try:
            t_rs = time.time()
            decoded_headers = bytes(rsc.decode(rx_bytes)[0])
            end_t_rs = time.time()
        except ReedSolomonError as e:
            raise ValueError(f"Header RS decoding failed: {e}")
    else:
        decoded_headers = rx_bytes
    print(f"[decode] Decoded headers length: {len(decoded_headers)}")
    print(f"[decode] First 10 decoded header bytes: {decoded_headers[:10].hex()}")
    print(f"[decode] Last 10 decoded header bytes: {decoded_headers[-10:].hex()}")
    t6 = time.time()
    # print(f"[perf] RS decode: {end_t_rs - t_rs} sec")
    
    sos_index = decoded_headers.find(b'\xff\xda')
    if not (decoded_headers.startswith(b'\xff\xd8') and sos_index != -1):
        raise ValueError(f"Invalid JPEG headers: start={decoded_headers[:2].hex()}, sos_index={sos_index}")
    print(f"[decode] SOS marker found at byte index: {sos_index}")
    
    # Despread data
    protected_data = received_pm[data_start:data_end]
    print(f"[decode] Protected data length: {len(protected_data)} bits")
    if USE_PRBS:
        n_groups_data = len(protected_data) // CHIP_LENGTH_FOR_DATA
        chips_data = protected_data[:n_groups_data * CHIP_LENGTH_FOR_DATA].reshape(-1, CHIP_LENGTH_FOR_DATA)
        rx_bits_pm_data = np.dot(chips_data, DATA_PRBS) / CHIP_LENGTH_FOR_DATA
        rx_bits_data = ((np.sign(rx_bits_pm_data) + 1) / 2).astype(np.uint8)
    else:
        rx_bits_data = ((protected_data + 1) / 2).astype(np.uint8)
    print(f"[decode] Despread data bits length: {len(rx_bits_data)}")
    print(f"[decode] First 16 despread data bits: {rx_bits_data[:16].tolist()}")
    
    data_bytes = np.packbits(rx_bits_data).tobytes()
    fixed_data = fix_false_markers(data_bytes)
    print(f"[decode] Fixed data length: {len(fixed_data)}")
    t7 = time.time()
    print(f"[perf] fix markers: {t7 - t6} sec")
    
    result = jpg_build(decoded_headers, fixed_data)
    print(f"Total decode time: {time.time() - t0} sec")
    return result

if __name__ == "__main__":
    cap = cv2.VideoCapture(PATH_TO_VIDEO)
    frame_count = 0
    while cap.isOpened() and frame_count < 4:
        frame_count += 1
        success, frame = cap.read()
        if not success:
            break
        h, w = frame.shape[:2]
        print(f"Original: {w}×{h}")
        
        frame_proc = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        encode_param = [cv2.IMWRITE_JPEG_QUALITY, 30, cv2.IMWRITE_JPEG_RST_INTERVAL, 1]
        _, encoded_image = cv2.imencode(".jpg", frame_proc, encode_param)
        headers, compressed = jpg_parse(encoded_image.tobytes())
        
        try:
            frame_bin = encode_udp_to_frame(headers, compressed)
            cv2.imwrite(f"encoded_{frame_count}.png", frame_bin)
            print(f"Encoded frame {frame_count} saved.")
            noisy = frame_bin.copy()
            noise_mask = np.random.choice([0, 255], size=noisy.shape, p=[0.99, 0.01]).astype(np.uint8) # 3% bit flips
            noisy ^= noise_mask
            decoded_data = decode_frame_to_udp(noisy)
            with open(f"recovered_{frame_count}.jpg", "wb") as out_file:
                out_file.write(decoded_data)
            print(f"Decoded matches: {decoded_data == encoded_image.tobytes()}")
        
        except ValueError as e:
            print(f"Error: {e}")