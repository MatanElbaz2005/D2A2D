import numpy as np
try:
    from creedsolo import RSCodec, ReedSolomonError
except ImportError:
    print("cant find creedsolo, using reedsolo instead")
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
CHUNK_BYTES = 128
CHIP_LENGTH_FOR_HEADERS = 3
CHIP_LENGTH_FOR_DATA = 1
DATA_PRBS_POLY = [8, 2]
FORMAT_IMAGE = 'jpg'
MEMORY = [6]
G_MATRIX = [[0o133, 0o171]]
TB_LENGTH = 15
PATH_TO_VIDEO = r"C:\Users\matan\OneDrive\מסמכים\Matan\D2A2D\1572378-sd_960_540_24fps.mp4"

# PRBS flags
USE_PRBS_FOR_HEADERS = True
USE_PRBS_FOR_DATA = False

# RS flags
USE_RS_FOR_HEADERS = True
USE_RS_FOR_DATA = True

perp_rsc_time = time.time()
rsc = RSCodec(ECC_SYMBOLS)
print("preper rs took " + str(time.time() - perp_rsc_time))

# Sync patterns (Barker codes, ±1)
HEADERS_SYNC_PATTERN = np.array([1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1], dtype=np.int32)
DATA_SYNC_PATTERN = np.array([1, -1, 1, -1, 1, 1, -1, -1, 1, 1, 1, 1, 1], dtype=np.int32)
END_SYNC_PATTERN = np.array([-1, -1, -1, -1, -1, 1, 1, -1, -1, 1, -1, 1, -1, 1, 1, -1, 1, -1, -1, 1], dtype=np.int32)  # Extended pattern

# PRBS for spreading (if enabled)
HEADERS_PRBS = generate_prbs(CHIP_LENGTH_FOR_HEADERS, DATA_PRBS_POLY, 3) if USE_PRBS_FOR_HEADERS else None
DATA_PRBS = generate_prbs(CHIP_LENGTH_FOR_DATA, DATA_PRBS_POLY, 3) if USE_PRBS_FOR_DATA else None

def encode_udp_to_frame(headers: bytes, data: bytes) -> np.ndarray:
    start_time = time.time()
    
    if USE_RS_FOR_HEADERS:
        start_rs_headers_encode = time.time()
        coded_headers = rsc.encode(bytearray(headers))
        print("rs encode for headers took " + str(time.time() - start_rs_headers_encode))
    else:
        coded_headers = headers
    
    header_bits = np.unpackbits(np.frombuffer(coded_headers, dtype=np.uint8))
    header_bits_pm = header_bits.astype(np.int8) * 2 - 1
    
    if USE_PRBS_FOR_HEADERS:
        repeated_bits = np.repeat(header_bits_pm, CHIP_LENGTH_FOR_HEADERS)
        tiled_prbs = np.tile(HEADERS_PRBS, len(header_bits))
        protected_headers = repeated_bits * tiled_prbs
    else:
        mapping = {0: np.array([-1, 1, -1]), 1: np.array([1, -1, 1])}
        protected_headers = np.concatenate([mapping[bit] for bit in header_bits])
    
    if USE_RS_FOR_DATA:
        start_rs_data_encode = time.time()
        coded_blocks = []
        for i in range(0, len(data), CHUNK_BYTES):
            blk = data[i:i+CHUNK_BYTES]
            coded_blocks.append(rsc.encode(bytearray(blk))) 
        coded_data = b"".join(coded_blocks)
        print("rs encode (chunked) took " + str(time.time() - start_rs_data_encode))
    else:
        coded_data = data

    
    data_bits = np.unpackbits(np.frombuffer(coded_data, dtype=np.uint8))
    data_bits_pm = data_bits.astype(np.int8) * 2 - 1
    
    if USE_PRBS_FOR_DATA:
        repeated_data_bits = np.repeat(data_bits_pm, CHIP_LENGTH_FOR_DATA)
        tiled_prbs = np.tile(DATA_PRBS, len(data_bits))
        protected_data = repeated_data_bits * tiled_prbs
    else:
        protected_data = data_bits_pm

    full_stream = np.concatenate((HEADERS_SYNC_PATTERN, protected_headers, DATA_SYNC_PATTERN, protected_data, END_SYNC_PATTERN))
    print(f"[encode] Full stream length: {len(full_stream)} bits")
    
    total_pixels = FRAME_WIDTH * FRAME_HEIGHT
    if len(full_stream) > total_pixels:
        raise ValueError(f"Data too large: {len(full_stream)} bits > {total_pixels} pixels")
    
    full_u8 = (((full_stream + 1) // 2).astype(np.uint8)) * 255
    if full_u8.size < total_pixels:
        full_u8 = np.pad(full_u8, (0, total_pixels - full_u8.size), mode='constant')
    frame = full_u8.reshape((FRAME_HEIGHT, FRAME_WIDTH))
    
    print(f"[encode] Frame shape: {frame.shape}")
    print(f"Encode took: {time.time() - start_time} sec")
    return frame

def decode_frame_to_udp(frame: np.ndarray, corr_threshold: float = 0.8) -> bytes:
    t0 = time.time()
    if frame.shape != (FRAME_HEIGHT, FRAME_WIDTH):
        raise ValueError(f"Frame size mismatch: expected {FRAME_HEIGHT}x{FRAME_WIDTH}")
    
    received_pm = 2 * (frame.ravel() > 127).astype(np.int32) - 1
    t1 = time.time()
    
    corr_headers = signal.correlate(received_pm, HEADERS_SYNC_PATTERN, mode='valid') / len(HEADERS_SYNC_PATTERN)
    corr_data = signal.correlate(received_pm, DATA_SYNC_PATTERN, mode='valid') / len(DATA_SYNC_PATTERN)
    corr_end = signal.correlate(received_pm, END_SYNC_PATTERN, mode='valid') / len(END_SYNC_PATTERN)
    t2 = time.time()
    
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
    
    t4 = time.time()
    rx_bytes = np.packbits(rx_bits_headers).tobytes()
    t5 = time.time()
    
    if USE_RS_FOR_HEADERS:
        try:
            t_rs_headers = time.time()
            decoded_headers = bytes(rsc.decode(bytearray(rx_bytes))[0])
            end_t_rs_headers = time.time()
            print("rs decode for headers time " + str(end_t_rs_headers - t_rs_headers))
        except ReedSolomonError as e:
            raise ValueError(f"Header RS decoding failed: {e}")
    else:
        decoded_headers = rx_bytes
    t6 = time.time()
    # print(f"[perf] RS decode: {end_t_rs - t_rs} sec")
    
    sos_index = decoded_headers.find(b'\xff\xda')
    if not (decoded_headers.startswith(b'\xff\xd8') and sos_index != -1):
        raise ValueError(f"Invalid JPEG headers: start={decoded_headers[:2].hex()}, sos_index={sos_index}")
    
    # Despread data
    protected_data = received_pm[data_start:data_end]
    if USE_PRBS_FOR_DATA:
        n_groups_data = len(protected_data) // CHIP_LENGTH_FOR_DATA
        chips_data = protected_data[:n_groups_data * CHIP_LENGTH_FOR_DATA].reshape(-1, CHIP_LENGTH_FOR_DATA)
        rx_bits_pm_data = np.dot(chips_data, DATA_PRBS) / CHIP_LENGTH_FOR_DATA
        rx_bits_data = ((np.sign(rx_bits_pm_data) + 1) / 2).astype(np.uint8)
    else:
        rx_bits_data = ((protected_data + 1) / 2).astype(np.uint8)
    
    data_bytes = np.packbits(rx_bits_data).tobytes()
    
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
                decoded_chunks.append(bytes([0]) * CHUNK_BYTES)
            i += N
        # last (possibly shorter) block
        if i < len(data_bytes):
            blk = data_bytes[i:]
            try:
                decoded_chunks.append(bytes(rsc.decode(bytearray(blk))[0]))
            except ReedSolomonError:
                k_last = max(0, len(blk) - ECC_SYMBOLS)
                decoded_chunks.append(bytes([0]) * k_last)
        decoded_data = b"".join(decoded_chunks)
        print("rs decode (chunked) took " + str(time.time() - t_rs_data))
    else:
        decoded_data = data_bytes
    
    fixed_data = fix_false_markers(decoded_data)
    t7 = time.time()
    
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
        encode_param = [(cv2.IMWRITE_JPEG_QUALITY), 75, cv2.IMWRITE_JPEG_RST_INTERVAL, 1]
        _, encoded_image = cv2.imencode(".jpg", frame_proc, encode_param)
        headers, compressed = jpg_parse(encoded_image.tobytes())
        
        try:
            # encode
            frame  = encode_udp_to_frame(headers, compressed)
            
            # save the encoded frame
            cv2.imwrite(f"encoded_{frame_count}.png", frame)
            print(f"Encoded frame {frame_count} saved.")

            # add noise
            p = 0.01
            noisy = frame.copy()
            flip_mask = (np.random.rand(*noisy.shape) < p)
            noisy[flip_mask] ^= np.uint8(255)

            # decode
            decoded_data = decode_frame_to_udp(noisy)
            with open(f"recovered_{frame_count}.jpg", "wb") as out_file:
                out_file.write(decoded_data)
            print(f"Decoded matches: {decoded_data == encoded_image.tobytes()}")
        
        except ValueError as e:
            print(f"Error: {e}")