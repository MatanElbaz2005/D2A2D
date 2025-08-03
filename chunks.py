import numpy as np
from reedsolo import RSCodec, ReedSolomonError
from scipy import signal
import cv2
from webp_encdec import encode_frame_to_webp, decode_webp_to_frame
import time

# Config
FRAME_W = 720
FRAME_H = 480  # NTSC; use 576 for PAL
ECC_SYMBOLS = 32
SYNC_PRBS_LENGTH = 63
SYNC_PRBS_POLY = [6, 5]
CHIP_LENGTH = 3
DATA_PRBS_POLY = [2, 1]
FORMAT_IMAGE = 'jpg'  # can be WEBP or JPG
MEMORY = [6]
G_MATRIX = [[0o133, 0o171]]
TB_DEPTH = 15
INTERLEAVER_ROWS = 8
PATH_TO_VIDEO = r"C:\Users\matan\OneDrive\מסמכים\Matan\D2A2D\1572378-sd_960_540_24fps.mp4"
USE_INTERLEAVER = False
SYNC_BYTES = b'\xD3\xA1\xCF\x55'
SYNC_BITS  = np.unpackbits(np.frombuffer(SYNC_BYTES, np.uint8))

TILE_ROWS = 5
TILE_COLS = 5
TILE_COUNT = TILE_ROWS * TILE_COLS
TILE_W    = FRAME_W  // TILE_COLS   # 720 // 5 = 144
TILE_H    = FRAME_H // TILE_ROWS   # 480 // 5 =  96

def tile_to_jpeg(tile: np.ndarray, q: int = 30) -> bytes:
    ok, buf = cv2.imencode(".jpg", tile, [cv2.IMWRITE_JPEG_QUALITY, q])
    if not ok:
        raise ValueError("cv2.imencode failed")
    return buf.tobytes()

def jpeg_to_bgr(data: bytes) -> np.ndarray | None:
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    return img



def encode_tiles_stream(tiles: list[bytes]) -> np.ndarray:
    """
    tiles : list of 25 JPEG-encoded byte-arrays (one per tile)
    returns: 720×480 mono image – 0 = black, 255 = white.
    """
    bitstream = SYNC_BITS.tolist()
    length_placeholder = [0]*32 # BEFORE we know the size
    bitstream.extend(length_placeholder)
    payload_start = len(bitstream) # remember where payload really starts

    for tid, raw in enumerate(tiles):
        size   = len(raw)
        header = bytes([tid]) + size.to_bytes(2, 'big')
        payload_bits = np.unpackbits(np.frombuffer(header + raw, np.uint8))
        print(f"[ENC] tile {tid:02d}: {size} B  ->  {payload_bits.size} bits")
        bitstream.extend(payload_bits.tolist())

    payload_len = len(bitstream) - payload_start      # number of bits of real payload
    len_bytes = payload_len.to_bytes(4, 'big')        # 32-bit big-endian length
    len_bits  = np.unpackbits(np.frombuffer(len_bytes, np.uint8))
    bitstream[payload_start-32 : payload_start] = len_bits   # overwrite placeholder
    
    total_bits = FRAME_W * FRAME_H
    print(f"[ENC] stream bits: {len(bitstream)} / canvas: {total_bits}") 
    if len(bitstream) > total_bits:
        raise ValueError("Frame overflow - increase compression or reduce tiles.")
    bitstream.extend([0] * (total_bits - len(bitstream)))

    frame = (np.array(bitstream, np.uint8)
               .reshape(FRAME_H, FRAME_W) * 255)
    return frame


def decode_stream_to_tiles(frame: np.ndarray) -> list[bytes | None]:
    """Extract list[25] (bytes or None) from a received mono frame."""
    bits = (frame.ravel() > 127).astype(np.uint8)

    for pos in range(0, bits.size - SYNC_BITS.size + 1, 8):
        if np.array_equal(bits[pos:pos+SYNC_BITS.size], SYNC_BITS):
            break
    else:
        raise ValueError("Sync not found")
    idx  = pos + SYNC_BITS.size
    total_len_bits = bits[idx : idx+32]  # read the 32-bit length
    payload_len    = int.from_bytes(np.packbits(total_len_bits), 'big')
    idx += 32
    end_idx = idx + payload_len  # last bit that is still payload

    tiles = [None]*TILE_COUNT
    while idx + 24 <= end_idx: 
        header_bits  = bits[idx : idx+24]               
        idx += 24   
        hdr_bytes = np.packbits(header_bits)                    
        tid = hdr_bytes[0]                       
        size = int.from_bytes(hdr_bytes[1:3], 'big')
        nbits = size * 8
        if idx + nbits > end_idx:
            break                          

        byte_arr = np.packbits(bits[idx:idx+nbits])[:size]
        idx += nbits

        if 0 <= tid < TILE_COUNT:
            tiles[tid] = byte_arr       
        
    print(f"[DEC] recovered: {sum(t is not None for t in tiles)}   "
      f"missing: {sum(t is None  for t in tiles)}") 
    
    return tiles


# Example usage
if __name__ == "__main__":
    cap = cv2.VideoCapture(PATH_TO_VIDEO)
    frame_count = 0
    while cap.isOpened() and frame_count < 4:
        frame_count += 1
        success, frame = cap.read()
        if not success:
            break
        # the original frame size
        h, w = frame.shape[:2]
        print("--------------------------------")
        print(f"original: {w}×{h}")
        print("---------------------------------")

        # resize to 320x240 if larger
        TARGET_W, TARGET_H = 720, 480
        frame_proc = cv2.resize(frame, (TARGET_W, TARGET_H))
        try:
            tiles_bytes = []
            for tr in range(TILE_ROWS):
                for tc in range(TILE_COLS):
                    y0, y1 = tr*TILE_H, (tr+1)*TILE_H
                    x0, x1 = tc*TILE_W, (tc+1)*TILE_W
                    tile   = frame_proc[y0:y1, x0:x1]
                    tiles_bytes.append(tile_to_jpeg(tile, q=30))

            frame_bin = encode_tiles_stream(tiles_bytes)
            cv2.imwrite(f"encoded_{frame_count}.png", frame_bin)
            print(f"Encoded frame {frame_count} saved.")
            noisy = frame_bin.copy()
            noise_mask = np.random.choice([0, 255], size=noisy.shape,
                                            p=[0.97, 0.03]).astype(np.uint8) # 3% bit flips
            noisy ^= noise_mask
            tiles_dec = decode_stream_to_tiles(frame_bin)
            print("[MAIN] decoded lengths:",
                [len(b) if b is not None else 0 for b in tiles_dec])
            restored  = np.zeros_like(frame_proc)

            for tid, data in enumerate(tiles_dec):
                if data is None:
                    continue
                img = jpeg_to_bgr(data)
                if img is None:
                    continue
                r, c = divmod(tid, TILE_COLS)
                y0, y1 = r*TILE_H, (r+1)*TILE_H
                x0, x1 = c*TILE_W, (c+1)*TILE_W
                restored[y0:y1, x0:x1] = cv2.resize(img, (TILE_W, TILE_H))
            cv2.imwrite(f"restored_{frame_count}.jpg", restored)


        except ValueError as e:
            print(f"Error: {e}")