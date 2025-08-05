import numpy as np
from reedsolo import RSCodec, ReedSolomonError
from scipy import signal
import cv2
from webp_encdec import encode_frame_to_webp, decode_webp_to_frame
import time
from helpers import generate_prbs

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
PATH_TO_VIDEO = r"/home/pi/Documents/matan/code/D2A2D/1572378-sd_960_540_24fps.mp4"
SYNC_BYTES = b'\xD3\xA1\xCF\x55'
SYNC_BITS  = np.unpackbits(np.frombuffer(SYNC_BYTES, np.uint8))

#flags
USE_RS = False
USE_PRBS = False
USE_INTERLEAVER = False

TILE_ROWS = 5
TILE_COLS = 5  
TILE_COUNT = TILE_ROWS * TILE_COLS
TILE_W    = FRAME_W  // TILE_COLS   # 720 // 5 = 144
TILE_H    = FRAME_H // TILE_ROWS   # 480 // 5 =  96


if USE_PRBS:
    DATA_PRBS = generate_prbs(CHIP_LENGTH, DATA_PRBS_POLY)

def tile_encode(tile: np.ndarray, quality: int = 30) -> bytes:
    """encode one 144 × 96 tile to JPEG / WebP according to FORMAT_IMAGE"""
    if FORMAT_IMAGE.lower() == "webp":
        return encode_frame_to_webp(tile, quality=quality)
    else:
        ok, buf = cv2.imencode(".jpg", tile,
                               [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            raise ValueError("cv2.imencode failed")
        return buf.tobytes()

def tile_decode(data: bytes) -> np.ndarray | None:
    if FORMAT_IMAGE.lower() == "webp":
        return decode_webp_to_frame(data)
    else:
        return cv2.imdecode(np.frombuffer(data, np.uint8),
                            cv2.IMREAD_COLOR)



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
        if USE_RS:
            raw_coded = RSCodec(ECC_SYMBOLS).encode(raw)
        else:
            raw_coded = raw

        size   = len(raw_coded)
        header = bytes([tid]) + size.to_bytes(2, 'big')
        payload_bits = np.unpackbits(np.frombuffer(header + raw_coded, np.uint8))

        if USE_PRBS:
            spread = np.repeat(payload_bits*2-1, CHIP_LENGTH) * np.tile(DATA_PRBS, payload_bits.size)
            bitstream.extend(((spread+1)//2).astype(np.uint8).tolist())
        else:
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

    # Signed ±1 vectors
    rx_pm   = 2*bits.astype(np.int16) - 1 # {0,1}→{-1,+1}
    sync_pm = 2*SYNC_BITS.astype(np.int16) - 1

    # correlation
    corr = np.correlate(rx_pm, sync_pm, mode='valid').astype(np.int32)

    pos  = int(corr.argmax())
    score = corr[pos] / SYNC_BITS.size
    print(f"[DEC] best-corr pos={pos}  score={score:.3f}")

    if score < 0.7:
        raise ValueError("Sync not found – correlation too low")

    idx = pos + SYNC_BITS.size
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
            print("[DEC] truncated payload – expected", nbits, "bits but only", end_idx-idx, "left")
            break                          

        if USE_PRBS:
            spread_bits = bits[idx:idx+nbits*CHIP_LENGTH]*2 - 1
            despread   = spread_bits.reshape(-1, CHIP_LENGTH) @ DATA_PRBS
            despread   = ((np.sign(despread) + 1) // 2).astype(np.uint8)
            byte_arr   = np.packbits(despread)[:size]
            idx       += nbits*CHIP_LENGTH
        else:
            byte_arr = np.packbits(bits[idx:idx+nbits])[:size]
            idx     += nbits

        if USE_RS:
            try:
                byte_arr = RSCodec(ECC_SYMBOLS).decode(byte_arr)[0]
            except ReedSolomonError as e:
                print("[DEC] RS decode failed:", e)
                byte_arr = None
                
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
            total_encode_time = 0
            tiles_bytes = []
            for tr in range(TILE_ROWS):
                for tc in range(TILE_COLS):
                    y0, y1 = tr*TILE_H, (tr+1)*TILE_H
                    x0, x1 = tc*TILE_W, (tc+1)*TILE_W
                    tile   = frame_proc[y0:y1, x0:x1]
                    start_encode = time.time()
                    tiles_bytes.append(tile_encode(tile, quality=30))
                    end_encode = time.time()
                    total_encode_time += end_encode - start_encode

            print(f"[MAIN] encoded {len(tiles_bytes)} tiles in {total_encode_time} sec for {FORMAT_IMAGE} format")
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
            
            total_decode_time = 0
            for tid, data in enumerate(tiles_dec):
                if data is None:
                    continue
                start_decode = time.time()
                img = tile_decode(data)
                end_decode = time.time()
                total_decode_time += end_decode - start_decode
                if img is None:
                    continue
                r, c = divmod(tid, TILE_COLS)
                y0, y1 = r*TILE_H, (r+1)*TILE_H
                x0, x1 = c*TILE_W, (c+1)*TILE_W
                restored[y0:y1, x0:x1] = cv2.resize(img, (TILE_W, TILE_H))
            print(f"[MAIN] decoded {len(tiles_dec)} tiles in {total_decode_time} sec for {FORMAT_IMAGE} format")
            cv2.imwrite(f"restored_{frame_count}.jpg", restored)


        except ValueError as e:
            print(f"Error: {e}")