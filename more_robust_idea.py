import numpy as np
from reedsolo import RSCodec, ReedSolomonError
from scipy import signal  # For correlation if needed

# Config
FRAME_WIDTH = 720
FRAME_HEIGHT = 480
ECC_SYMBOLS = 32
SYNC_PRBS_LENGTH = 255  # Frame sync PRBS
SYNC_PRBS_POLY = [8, 6, 5, 4]
CHIP_LENGTH = 5  # Spreading factor; tune for rate vs resilience
DATA_PRBS_POLY = [3, 2]  # Short PRBS for data (degree 3: 7 chips max, but use 5)

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
    return 2 * prbs[:length] - 1  # ±1, truncated if needed

SYNC_PRBS = generate_prbs(SYNC_PRBS_LENGTH, SYNC_PRBS_POLY)
DATA_PRBS = generate_prbs(CHIP_LENGTH, DATA_PRBS_POLY)

def encode_udp_to_frame(udp_data: bytes) -> np.ndarray:
    rsc = RSCodec(ECC_SYMBOLS)
    coded_data = rsc.encode(udp_data)
    bit_stream = np.array([int(b) for byte in coded_data for b in f'{byte:08b}'])
    bits_pm = 2 * bit_stream - 1  # ±1 for spreading
    
    # Spread each bit
    spread_stream = np.repeat(bits_pm, CHIP_LENGTH) * np.tile(DATA_PRBS, len(bits_pm))
    
    # Prepend sync PRBS (unspread, for frame alignment)
    full_stream = np.concatenate((SYNC_PRBS, spread_stream))
    
    # Pad to frame size
    total_pixels = FRAME_WIDTH * FRAME_HEIGHT
    if len(full_stream) > total_pixels:
        raise ValueError("Data too large.")
    pad = np.zeros(total_pixels - len(full_stream), dtype=int)
    full_stream = np.concatenate((full_stream, pad))
    
    # Map to pixels: +1→255, -1→0
    frame = ((full_stream + 1) / 2 * 255).astype(np.uint8).reshape((FRAME_HEIGHT, FRAME_WIDTH))
    return frame

def decode_frame_to_udp(frame: np.ndarray, corr_threshold: float = 0.8) -> bytes:
    bit_stream = (frame.flatten() > 127).astype(int)  # Threshold to 0/1
    received_pm = 2 * bit_stream - 1  # ±1
    
    # Find sync via correlation
    corr = signal.correlate(received_pm, SYNC_PRBS, mode='valid') / SYNC_PRBS_LENGTH
    max_corr = np.max(corr)
    if max_corr < corr_threshold:
        raise ValueError(f"Sync not detected (max corr {max_corr})")
    sync_pos = np.argmax(corr)
    
    # Extract spread data after sync
    data_start = sync_pos + SYNC_PRBS_LENGTH
    spread_bits = received_pm[data_start:data_start + ((len(received_pm) - data_start) // CHIP_LENGTH * CHIP_LENGTH)]
    
    # Despread
    num_bits = len(spread_bits) // CHIP_LENGTH
    rx_bits_pm = np.zeros(num_bits)
    for i in range(num_bits):
        chip_slice = spread_bits[i*CHIP_LENGTH:(i+1)*CHIP_LENGTH]
        rx_bits_pm[i] = np.dot(chip_slice, DATA_PRBS) / CHIP_LENGTH
    rx_bits = (np.sign(rx_bits_pm) + 1) / 2  # Back to 0/1
    
    # Bits to bytes
    coded_data = bytearray(int(''.join(map(str, rx_bits[i:i+8].astype(int))), 2) for i in range(0, len(rx_bits) - len(rx_bits)%8, 8))
    
    # RS decode
    rsc = RSCodec(ECC_SYMBOLS)
    try:
        return bytes(rsc.decode(coded_data)[0])
    except ReedSolomonError as e:
        raise ValueError(f"Decoding failed: {e}")
