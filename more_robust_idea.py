import numpy as np
from reedsolo import RSCodec, ReedSolomonError
from scipy import signal

# Config
FRAME_WIDTH = 720
FRAME_HEIGHT = 480  # NTSC; use 576 for PAL
ECC_SYMBOLS = 32
SYNC_PRBS_LENGTH = 63
SYNC_PRBS_POLY = [6, 5]
CHIP_LENGTH = 3
DATA_PRBS_POLY = [2, 1]

def generate_prbs(length: int, poly: list[int]) -> np.ndarray:
    degree = poly[0]
    state = (1 << degree) - 1
    prbs = np.zeros(length, dtype=int)
    for i in range(length):
        bit = state & 1
        prbs[i] = bit
        feedback = 0
        for tap in poly[1:]:
            feedback ^= (state >> (degree - tap)) & 1
        state = (state >> 1) | (feedback << (degree - 1))
    return 2 * prbs[:length] - 1  # ±1

SYNC_PRBS = generate_prbs(SYNC_PRBS_LENGTH, SYNC_PRBS_POLY)
DATA_PRBS = generate_prbs(CHIP_LENGTH, DATA_PRBS_POLY)

def encode_udp_to_frame(udp_data: bytes) -> np.ndarray:
    """
    Encode UDP packet into a black/white frame with vectorized spread-spectrum.
    """
    rsc = RSCodec(ECC_SYMBOLS)
    coded_data = rsc.encode(udp_data)
    bit_stream = np.unpackbits(np.frombuffer(coded_data, dtype=np.uint8))
    bits_pm = 2 * bit_stream - 1  # ±1
    # Vectorized spreading: repeat bits and multiply by tiled PRBS
    repeated_bits = np.repeat(bits_pm, CHIP_LENGTH)
    tiled_prbs = np.tile(DATA_PRBS, len(bits_pm))
    spread_stream = repeated_bits * tiled_prbs
    full_stream = np.concatenate((SYNC_PRBS, spread_stream))
    total_pixels = FRAME_WIDTH * FRAME_HEIGHT
    if len(full_stream) > total_pixels:
        raise ValueError(f"Data too large: {len(full_stream)} bits > {total_pixels} pixels")
    # Vectorized padding
    full_stream = np.pad(full_stream, (0, total_pixels - len(full_stream)))
    # Map to pixels: +1→255, -1→0
    frame = ((full_stream + 1) / 2 * 255).astype(np.uint8).reshape((FRAME_HEIGHT, FRAME_WIDTH))
    return frame

def decode_frame_to_udp(frame: np.ndarray, corr_threshold: float = 0.8) -> bytes:
    """
    Decode a noisy frame with vectorized despreading.
    """
    if frame.shape != (FRAME_HEIGHT, FRAME_WIDTH):
        raise ValueError(f"Frame size mismatch: expected {FRAME_HEIGHT}x{FRAME_WIDTH}")
    received_pm = 2 * (frame.ravel() > 127).astype(np.int32) - 1  # ±1, int32 for dot
    corr = signal.correlate(received_pm, SYNC_PRBS, mode='valid') / SYNC_PRBS_LENGTH
    max_corr = np.max(corr)
    if max_corr < corr_threshold:
        raise ValueError(f"Sync not detected (max corr {max_corr})")
    sync_pos = np.argmax(corr)
    data_start = sync_pos + SYNC_PRBS_LENGTH
    # Truncate to multiple of CHIP_LENGTH
    spread_len = ((len(received_pm) - data_start) // CHIP_LENGTH * CHIP_LENGTH)
    spread_bits = received_pm[data_start:data_start + spread_len]
    # Vectorized despreading: reshape and matrix multiply
    chips = spread_bits.reshape(-1, CHIP_LENGTH)
    rx_bits_pm = np.dot(chips, DATA_PRBS) / CHIP_LENGTH
    rx_bits = ((np.sign(rx_bits_pm) + 1) / 2).astype(np.uint8)
    # Pack bits to bytes (vectorized)
    coded_data = np.packbits(rx_bits)
    rsc = RSCodec(ECC_SYMBOLS)
    try:
        return bytes(rsc.decode(coded_data)[0])
    except ReedSolomonError as e:
        raise ValueError(f"Decoding failed: {e}")

# Example usage
if __name__ == "__main__":
    sample_udp_data = b'\x00\x01\x02\x03' * 8192  # 32 KB, but will raise error if too large
    try:
        frame = encode_udp_to_frame(sample_udp_data)
        decoded_data = decode_frame_to_udp(frame)
        print(f"Decoded matches: {decoded_data == sample_udp_data}")
    except ValueError as e:
        print(f"Error: {e}")
