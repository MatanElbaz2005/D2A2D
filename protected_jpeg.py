import struct
from typing import Tuple

def split_jpeg(data: bytes) -> Tuple[bytes, bytes]:
    """
    Split a JPEG byte array into header (critical/small segments up to SOS) and data (bulk compressed content).
    Returns: (header_bytes, data_bytes)
    """
    header = bytearray()
    i = 0
    data_len = len(data)

    while i < data_len - 1:
        if data[i] == 0xFF:
            marker = (data[i] << 8) | data[i + 1]
            header.extend(data[i:i + 2])
            i += 2

            # Standalone markers (e.g., SOI)
            if marker in (0xFFD8, 0xFFD9):
                continue  # EOI shouldn't be here, but handled later

            if marker >= 0xFFD0 and marker <= 0xFFD7:  # RST in header (rare)
                continue

            if i + 2 > data_len:
                raise ValueError("Incomplete JPEG segment")

            length = struct.unpack(">H", data[i:i + 2])[0]
            header.extend(data[i:i + 2])
            i += 2

            if i + length - 2 > data_len:
                raise ValueError("Incomplete JPEG segment data")

            segment_data = data[i:i + length - 2]
            header.extend(segment_data)
            i += length - 2

            if marker == 0xFFDA:  # SOS found, now extract compressed data
                compressed_start = i
                # Scan for EOI, skipping escaped 0xFF00 and ignoring RST markers
                while i < data_len - 1:
                    if data[i] == 0xFF:
                        next_byte = data[i + 1]
                        if next_byte == 0x00:
                            i += 2
                            continue
                        elif next_byte == 0xD9:  # EOI
                            compressed_end = i
                            break
                        elif 0xD0 <= next_byte <= 0xD7:  # RST
                            i += 2
                            continue
                        else:
                            # Unexpected marker; assume continue for robustness
                            i += 2
                            continue
                    else:
                        i += 1
                else:
                    raise ValueError("No EOI found in JPEG")

                data_bytes = data[compressed_start:compressed_end]
                return bytes(header), data_bytes

        else:
            raise ValueError("Invalid JPEG: Non-marker byte in header")

    raise ValueError("No SOS segment found in JPEG")

def merge_jpeg(header: bytes, data: bytes) -> bytes:
    """
    Merge the header and data back into a complete JPEG byte array by appending EOI.
    """
    return header + data + b'\xff\xd9'


def fix_false_markers(compressed: bytes) -> bytes:
    ba = bytearray(compressed)
    pos = 0
    valid_next = {0x00} | set(range(0xd0, 0xd8))
    while True:
        pos = ba.find(255, pos)
        if pos == -1 or pos >= len(ba) - 1:
            break
        next_byte = ba[pos + 1]
        if next_byte in valid_next:
            pos += 2
            continue
        else:
            ba[pos + 1] = 0x00
            pos += 2
    return bytes(ba)

# def encode_udp_to_frame(headers: bytes, data: bytes) -> np.ndarray:
    
#     print(f"[encode] Header size: {len(headers)} bytes")
#     print(f"[encode] First 10 header bytes: {headers[:10].hex()}")
#     print(f"[encode] Last 10 header bytes: {headers[-10:].hex()}")
    
#     coded_headers = headers
#     print(f"[encode] No RS, header size: {len(coded_headers)} bytes")
    
#     header_bits = np.unpackbits(np.frombuffer(coded_headers, dtype=np.uint8))
#     print(f"[encode] Header bits length: {len(header_bits)}")
#     header_bits_pm = 2.0 * header_bits - 1
    
#     mapping = {0: np.array([-1, 1, -1]), 1: np.array([1, -1, 1])}
#     protected_headers = np.concatenate([mapping[bit] for bit in header_bits])
#     print(f"[encode] Using 3-bit mapping, protected headers length: {len(protected_headers)} bits")
    
#     data_bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
#     print(f"[encode] Data bits length: {len(data_bits)}")

#     data_bits_pm = 2.0 * data_bits - 1
#     protected_data = data_bits_pm

#     return HEADERS_SYNC_PATTERN, protected_headers, DATA_SYNC_PATTERN, protected_data, END_SYNC_PATTERN

# def decode_frame_to_udp(frame: np.ndarray, corr_threshold: float = 0.8) -> bytes:
#     t0 = time.time()
#     if frame.shape != (FRAME_HEIGHT, FRAME_WIDTH):
#         raise ValueError(f"Frame size mismatch: expected {FRAME_HEIGHT}x{FRAME_WIDTH}")
    
#     received_pm = 2 * (frame.ravel() > 127).astype(np.int32) - 1
#     t1 = time.time()
#     print(f"[perf] threshold→±1: {t1 - t0} sec")
    
#     corr_headers = signal.correlate(received_pm, HEADERS_SYNC_PATTERN, mode='valid') / len(HEADERS_SYNC_PATTERN)
#     corr_data = signal.correlate(received_pm, DATA_SYNC_PATTERN, mode='valid') / len(DATA_SYNC_PATTERN)
#     corr_end = signal.correlate(received_pm, END_SYNC_PATTERN, mode='valid') / len(END_SYNC_PATTERN)
#     t2 = time.time()
#     print(f"[perf] correlations: {t2 - t1} sec")
#     print(f"[decode] Header sync max corr: {np.max(corr_headers)} at pos {np.argmax(corr_headers)}")
#     print(f"[decode] Data sync max corr: {np.max(corr_data)} at pos {np.argmax(corr_data)}")
#     print(f"[decode] End sync max corr: {np.max(corr_end)} at pos {np.argmax(corr_end)}")
    
#     if np.max(corr_headers) < corr_threshold or np.max(corr_data) < corr_threshold or np.max(corr_end) < corr_threshold:
#         raise ValueError(f"Sync not detected: headers={np.max(corr_headers)}, data={np.max(corr_data)}, end={np.max(corr_end)}")
    
#     headers_start = np.argmax(corr_headers) + len(HEADERS_SYNC_PATTERN)
#     data_start = np.argmax(corr_data) + len(DATA_SYNC_PATTERN)
#     data_end = np.argmax(corr_end)
    
#     if not (headers_start < data_start < data_end):
#         raise ValueError(f"Invalid sync pattern order: headers_start={headers_start}, data_start={data_start}, data_end={data_end}")
#     print(f"[decode] Headers range: {headers_start}:{data_start - len(DATA_SYNC_PATTERN)}")
#     print(f"[decode] Data range: {data_start}:{data_end}")
    
#     expected_data_bits = (data_end - data_start) * 8  # Approximate based on pixel range
#     print(f"[decode] Expected data bits (approx): {expected_data_bits}")
    
#     protected_headers = received_pm[headers_start:data_start - len(DATA_SYNC_PATTERN)]
#     print(f"[decode] Protected headers length: {len(protected_headers)} bits")
#     t3 = time.time()
#     print(f"[perf] extraction: {t3 - t2} sec")
    

#     n_groups = len(protected_headers) // 3
#     chips = protected_headers[:n_groups * 3].reshape(-1, 3)
#     patterns = np.array([[-1, 1, -1], [1, -1, 1]], dtype=np.int32)
#     corr = np.dot(chips, patterns.T) / 3
#     rx_bits_headers = (np.argmax(corr, axis=1)).astype(np.uint8)
#     print(f"[decode] Despread bits length: {len(rx_bits_headers)}")
#     print(f"[decode] First 16 despread bits: {rx_bits_headers[:16].tolist()}")
#     print(f"[decode] Last 16 despread bits: {rx_bits_headers[-16:].tolist()}")
    
#     t4 = time.time()
#     print(f"[perf] despreading: {t4 - t3} sec")
    
#     rx_bytes = np.packbits(rx_bits_headers).tobytes()
#     print(f"[decode] Packed bytes length: {len(rx_bytes)}")
#     print(f"[decode] First 10 packed bytes: {rx_bytes[:10].hex()}")
#     print(f"[decode] Last 10 packed bytes: {rx_bytes[-10:].hex()}")
#     t5 = time.time()
#     print(f"[perf] packbits: {t5 - t4} sec")
    

#     decoded_headers = rx_bytes
#     print(f"[decode] Decoded headers length: {len(decoded_headers)}")
#     print(f"[decode] First 10 decoded header bytes: {decoded_headers[:10].hex()}")
#     print(f"[decode] Last 10 decoded header bytes: {decoded_headers[-10:].hex()}")
#     t6 = time.time()
    
#     sos_index = decoded_headers.find(b'\xff\xda')
#     if not (decoded_headers.startswith(b'\xff\xd8') and sos_index != -1):
#         raise ValueError(f"Invalid JPEG headers: start={decoded_headers[:2].hex()}, sos_index={sos_index}")
#     print(f"[decode] SOS marker found at byte index: {sos_index}")
    
#     # Despread data
#     protected_data = received_pm[data_start:data_end]
#     print(f"[decode] Protected data length: {len(protected_data)} bits")
#     rx_bits_data = ((protected_data + 1) / 2).astype(np.uint8)
#     print(f"[decode] Despread data bits length: {len(rx_bits_data)}")
#     print(f"[decode] First 16 despread data bits: {rx_bits_data[:16].tolist()}")
    
#     data_bytes = np.packbits(rx_bits_data).tobytes()
#     return data_bytes, decoded_headers
#     # fixed_data = fix_false_markers(data_bytes)
#     # print(f"[decode] Fixed data length: {len(fixed_data)}")
#     # t7 = time.time()
#     # print(f"[perf] fix markers: {t7 - t6} sec")
    
#     # result = jpg_build(decoded_headers, fixed_data)
#     # print(f"Total decode time: {time.time() - t0} sec")
#     # return result

# cap = cv2.VideoCapture(PATH_TO_VIDEO)
# frame_count = 0
# while cap.isOpened() and frame_count < 4:
#     frame_count += 1
#     success, frame = cap.read()
#     if not success:
#         break
#     h, w = frame.shape[:2]
#     print(f"Original: {w}×{h}")
    
#     frame_proc = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

#     # Read the image
#     img = frame_proc.copy()

#     # Encode to JPEG binary using OpenCV
#     start_time = time.time()
#     params = [(cv2.IMWRITE_JPEG_QUALITY), 30, int(cv2.IMWRITE_JPEG_RST_INTERVAL), 1]  # Adjust interval; smaller = more resilient but larger file
#     success, jpg_bytes = cv2.imencode('.jpg', img, params)
#     if not success:
#         raise ValueError("Failed to encode image")

#     # Parse
#     start_parse = time.time()
#     header, compressed = jpg_parse(jpg_bytes.tobytes())
#     print("compressed length is: " + str(len(compressed)))
#     print("compressed first 10 vals are: " + str(compressed[:10].hex()))

#     # Encode to frame
#     HEADERS_SYNC_PATTERN, protected_headers, DATA_SYNC_PATTERN, protected_data, END_SYNC_PATTERN = encode_udp_to_frame(header, compressed)

#     full_stream = np.concatenate((
#         HEADERS_SYNC_PATTERN,
#         protected_headers,
#         DATA_SYNC_PATTERN,
#         protected_data,
#         END_SYNC_PATTERN
#     ))
#     print(f"[encode] Full stream length: {len(full_stream)} bits")
    
#     total_pixels = FRAME_WIDTH * FRAME_HEIGHT
#     if len(full_stream) > total_pixels:
#         raise ValueError(f"Data too large: {len(full_stream)} bits > {total_pixels} pixels")
    
#     full_stream = ((full_stream + 1) / 2 * 255).astype(np.uint8)
#     full_stream = np.pad(full_stream, (0, total_pixels - len(full_stream)), 'constant')
#     frame = full_stream.reshape((FRAME_HEIGHT, FRAME_WIDTH))

#     print(f"Parsed JPEG in {time.time() - start_parse} seconds")
#     noisy = protected_data.copy()
#     noisy = ((noisy + 1) / 2 * 255).astype(np.uint8)  # Maps -1 to 0, 1 to 255
#     noise_mask = np.random.choice([0, 255], size=noisy.shape, p=[0.99, 0.01]).astype(np.uint8)  # 1% bit flips
#     noisy ^= noise_mask


#     full_stream_final = np.concatenate((
#         HEADERS_SYNC_PATTERN,
#         protected_headers,
#         DATA_SYNC_PATTERN,
#         noisy,
#         END_SYNC_PATTERN
#     ))
    
#     print(f"[debug] Full stream final length before conversion: {len(full_stream_final)} bits")
#     total_pixels = FRAME_WIDTH * FRAME_HEIGHT
#     if len(full_stream_final) > total_pixels:
#         raise ValueError(f"Data too large: {len(full_stream_final)} bits > {total_pixels} pixels")

#     full_stream_final = ((full_stream_final + 1) / 2 * 255).astype(np.uint8)
#     full_stream_final = np.pad(full_stream_final, (0, total_pixels - len(full_stream_final)), 'constant')  # Pad to match frame size
#     frame_final = full_stream_final.reshape((FRAME_HEIGHT, FRAME_WIDTH))
#     cv2.imwrite(f"encoded_{frame_count}.png", frame_final)
#     print(f"Encoded frame {frame_count} saved.")


#     # Decode from frame
#     start_decode = time.time()
#     data_bytes, decoded_headers = decode_frame_to_udp(frame_final)
#     print("data_bytes length is: " + str(len(data_bytes)))
#     print("data_bytes first 10 vals are: " + str(data_bytes[:10].hex()))

#     start_fix = time.time()
#     fixed_compressed = fix_false_markers(data_bytes)
#     print(f"Fixed false markers in {time.time() - start_fix} seconds")


#     # Build back
#     start_rebuild = time.time()
#     rebuilt_bytes = jpg_build(decoded_headers, fixed_compressed)
#     print(f"Rebuilt JPEG in {time.time() - start_rebuild} seconds")

#     # Decode back with OpenCV to verify
#     rebuilt_img = cv2.imdecode(np.frombuffer(rebuilt_bytes, np.uint8), cv2.IMREAD_COLOR)
#     print(f"Total processing time: {time.time() - start_time} seconds")
#     if rebuilt_img is None:
#         raise ValueError("Failed to decode rebuilt image")
#     print("Rebuilt image shape:", rebuilt_img.shape)  # Should match original image shape

#     # Save the rebuilt image to verify visually
#     cv2.imwrite("rebuilt_image.jpg", rebuilt_img)
#     # print("Rebuilt image saved as 'rebuilt_image.jpg'")