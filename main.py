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
SYNC_PATTERN = '01' * 128  # 256-bit alternating sync pattern for frame alignment
SYNC_LENGTH = len(SYNC_PATTERN)

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
    if len(udp_data) > 65535:
        raise ValueError("UDP data too large (max 65535 bytes)")

    rsc = RSCodec(ecc_symbols)

    # Encode data with Reed-Solomon
    payload = len(udp_data).to_bytes(2, "big") + udp_data
    coded_data = rsc.encode(payload)
    
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
    sync_pos = bit_stream.find(SYNC_PATTERN)
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
            msg_len = int.from_bytes(decoded[:2], "big")
            if msg_len <= len(decoded) - 2:
                return bytes(decoded[2:2 + msg_len])
        except ReedSolomonError:
            continue

    raise ValueError("Reed-Solomon decoding failed or length field invalid")

# Example usage (for testing; assume you have a UDP packet and hardware for CVBS output/capture)
if __name__ == "__main__":
    # Simulated UDP packet with H.265 data (replace with real data)
    with open('lol.jpg', 'rb') as file:
        sample_udp_data = file.read()
    
    # Encode
    frame = encode_udp_to_frame(sample_udp_data)
    cv2.imwrite('encoded_frame.png', frame)  # Save for inspection (optional)
    
    # Simulate noise (for testing decoding robustness)
    noisy_frame = frame.copy()
    noise_mask = np.random.choice([0, 255], size=frame.shape, p=[0.9999, 0.0001])  # 0.01% bit flips
    noisy_frame = np.bitwise_xor(noisy_frame, noise_mask)
    
    # Decode
    try:
        decoded_data = decode_frame_to_udp(noisy_frame)
        print(f"Decoded data matches original: {decoded_data == sample_udp_data}")
        
        with open("recovered.jpg", "wb") as out_file:
            out_file.write(decoded_data)
            
    except ValueError as e:
        print(f"Decoding error: {e}")
