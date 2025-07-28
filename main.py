# Required libraries: Install with `pip install numpy opencv-python reedsolo`
# - numpy: For array operations
# - opencv-python: For potential frame output/display (can be replaced if needed)
# - reedsolo: For Reed-Solomon error correction

import numpy as np
import cv2  # Optional for displaying/saving frames; can remove if not needed
from reedsolo import RSCodec, ReedSolomonError

# Configuration parameters
FRAME_WIDTH = 720  # Typical NTSC/PAL width
FRAME_HEIGHT = 480  # NTSC height (use 576 for PAL)
ECC_SYMBOLS = 32   # Number of Reed-Solomon parity symbols (adjust for more/less error correction)
SYNC_PATTERN = '1111100110101' * 3
SYNC_LENGTH = len(SYNC_PATTERN)
PATH_TO_DATA = '1572378-sd_960_540_24fps.mp4'  # Path to sample data file (image or video frame)
MODE = 'video' # 'video' or 'single_image'

def find_sync_by_correlation(bit_stream: str, sync_pattern: str) -> int:
    def to_signed_bit_vector(bits: str) -> list[int]:
        return [1 if b == '1' else -1 for b in bits]

    sync_vector = to_signed_bit_vector(sync_pattern)
    window_size = len(sync_vector)
    max_score = float('-inf')
    best_position = -1

    for i in range(len(bit_stream) - window_size + 1):
        window = bit_stream[i:i + window_size]
        window_vector = to_signed_bit_vector(window)

        score = sum(s * w for s, w in zip(sync_vector, window_vector))
        if score > max_score:
            max_score = score
            best_position = i

    return best_position if max_score > 0.8 * window_size else -1


def encode_udp_to_frame(udp_data: bytes, width: int = FRAME_WIDTH, height: int = FRAME_HEIGHT, ecc_symbols: int = ECC_SYMBOLS) -> np.ndarray:
    """
    Encode a UDP packet (containing H.265 stream or other binary data) into a video frame.
    
    - Adds Reed-Solomon ECC for noise handling.
    - Prepends a sync pattern for decoding alignment.
    - Maps bits to black (0) or white (255) pixels in a grayscale frame.
    - Pads with zeros if data is smaller than frame size.
    - Raises error if data exceeds one frame (for streams, chunk input data across multiple calls).
    
    Args:
        udp_data: Bytes from UDP packet (e.g., H.265 payload + UDP headers).
        width: Frame width in pixels.
        height: Frame height in pixels.
        ecc_symbols: Number of RS parity symbols.
    
    Returns:
        numpy.ndarray: Grayscale video frame (uint8) ready for CVBS output.
    """
    # if len(udp_data) > 65535:
    #    raise ValueError("UDP data too large (max 65535 bytes)")

    rsc = RSCodec(ecc_symbols)

    # Encode data with Reed-Solomon
    payload = len(udp_data).to_bytes(4, "big") + udp_data
    print("Payload length:", len(payload), "bytes")
    coded_data = rsc.encode(payload)
    print("Coded data length:", len(coded_data), "bytes")
    
    # Convert to bit string
    bit_stream = ''.join(f'{byte:08b}' for byte in coded_data)
    
    # Prepend sync pattern
    bit_stream = SYNC_PATTERN + bit_stream
    
    # Check if it fits in one frame
    total_bits = len(bit_stream)
    total_pixels = width * height
    if total_bits > total_pixels:
        raise ValueError(f"Data too large for one frame ({total_bits} bits > {total_pixels} pixels). Chunk the UDP stream.")
    
    # Pad with zeros
    bit_stream += '0' * (total_pixels - total_bits)
    
    # Create the frame
    frame = np.zeros((height, width), dtype=np.uint8)
    idx = 0
    for y in range(height):
        for x in range(width):
            bit = int(bit_stream[idx])
            frame[y, x] = 255 if bit else 0
            idx += 1
    print(total_bits / 8, "bits encoded into frame")
    return frame

def decode_frame_to_udp(frame: np.ndarray, width: int = FRAME_WIDTH, height: int = FRAME_HEIGHT, ecc_symbols: int = ECC_SYMBOLS) -> bytes:
    """
    Decode a captured video frame back to the original UDP packet data.
    
    - Thresholds pixels to bits ( >127 -> 1, else 0).
    - Searches for sync pattern to align data.
    - Extracts bits, converts to bytes.
    - Applies Reed-Solomon decoding to correct errors from noise.
    
    Args:
        frame: numpy.ndarray grayscale frame (from CVBS capture).
        width: Frame width in pixels.
        height: Frame height in pixels.
        ecc_symbols: Number of RS parity symbols (must match encoding).
    
    Returns:
        bytes: Original UDP packet data (e.g., H.265 stream).
    
    Raises:
        ValueError: If sync not found or decoding fails.
    """
    if frame.shape != (height, width):
        raise ValueError(f"Frame size mismatch: expected {height}x{width}, got {frame.shape}")
    
    # Extract bits with thresholding
    bit_stream = ''
    for y in range(height):
        for x in range(width):
            pixel = frame[y, x]
            bit = '1' if pixel > 127 else '0'
            bit_stream += bit           
    
    # Find sync pattern
    sync_pos = find_sync_by_correlation(bit_stream, SYNC_PATTERN)
    if sync_pos == -1:
        raise ValueError("Sync pattern not found in frame")
    
    # Extract data bits after sync
    data_bits = bit_stream[sync_pos + SYNC_LENGTH:]
    
    # Convert bits to bytes (ignore trailing incomplete byte)
    coded_data = bytearray()
    for i in range(0, len(data_bits) - (len(data_bits) % 8), 8):
        byte_str = data_bits[i:i+8]
        coded_data.append(int(byte_str, 2))

    rsc = RSCodec(ecc_symbols)
    try:
        decoded, _, _ = rsc.decode(coded_data)
        msg_len = int.from_bytes(decoded[:4], "big")
        if msg_len <= len(decoded) - 4:
            return bytes(decoded[4:4 + msg_len])

    except ReedSolomonError:
        raise ValueError("Reed-Solomon decoding failed or length field invalid")

    raise ValueError("Reed-Solomon decoding failed or length field invalid")

# Example usage (for testing; assume you have a UDP packet and hardware for CVBS output/capture)
if __name__ == "__main__":
    # Simulated UDP packet with H.265 data (replace with real data)
    if MODE == 'single_image':
        with open('lol.jpg', 'rb') as file:
            sample_udp_data = file.read()
            
    elif MODE == 'video':
        cap = cv2.VideoCapture(PATH_TO_DATA)
        frame_count = 0
        while cap.isOpened() and frame_count < 4:
            frame_count += 1
            success, frame = cap.read()
            if not success:
                break
            # the original frame size
            h, w = frame.shape[:2]
            print(f"original: {w}×{h}")

            # resize to 320x240 if larger
            TARGET_W, TARGET_H = 720, 480
            frame_proc = cv2.resize(frame, (TARGET_W, TARGET_H))

            encode_param = [cv2.IMWRITE_JPEG_QUALITY, 75]
            _, encoded_image = cv2.imencode(".jpg", frame_proc, encode_param)
            full_bytes = encoded_image.tobytes()

            MAX_PAYLOAD = 37600
            chunks = [full_bytes[i:i + MAX_PAYLOAD]
                    for i in range(0, len(full_bytes), MAX_PAYLOAD)]
            parts = []

            for part_idx, chunk in enumerate(chunks):
                frame_bin = encode_udp_to_frame(chunk)
                cv2.imwrite(f"encoded_{frame_count}_{part_idx}.png", frame_bin)

                noisy = frame_bin.copy()
                noise_mask = np.random.choice([0, 255], size=noisy.shape,
                                            p=[0.999, 0.001]).astype(np.uint8)
                noisy ^= noise_mask



                try:
                    recovered_chunk = decode_frame_to_udp(noisy)
                    parts.append(recovered_chunk)
                    print(f"frame {frame_count}, chunk {part_idx}: OK ({len(recovered_chunk)} B)")
                except ValueError as err:
                    print(f"frame {frame_count}, chunk {part_idx}: {err}")

            if parts:
                full_bytes_recovered = b"".join(parts)
                img_arr = np.frombuffer(full_bytes_recovered, dtype=np.uint8)
                restored = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
                if restored is not None:
                    cv2.imwrite(f"restored_frame_{frame_count}.jpg", restored)
            
    if MODE == 'single_image':
        # Encode
        frame = encode_udp_to_frame(sample_udp_data)
        cv2.imwrite('encoded_frame.png', frame)  # Save for inspection (optional)
        
        # Simulate noise (for testing decoding robustness)
        noisy_frame = frame.copy()
        noise_mask = np.random.choice([0, 255], size=frame.shape, p=[0.999, 0.001])  # 1% bit flips
        noisy_frame = np.bitwise_xor(noisy_frame, noise_mask)
        
        # Decode
        try:
            decoded_data = decode_frame_to_udp(noisy_frame)
            print(f"Decoded data matches original: {decoded_data == sample_udp_data}")
            
            with open("recovered.jpg", "wb") as out_file:
                out_file.write(decoded_data)
                
        except ValueError as e:
            print(f"Decoding error: {e}")
