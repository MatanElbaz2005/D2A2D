import struct  # For unpacking lengths, though we use int.from_bytes
import cv2
import numpy as np
import time

def jpg_parse(jpg_bytes: bytes) -> tuple[bytes, bytes]:
    """
    Parses a JPEG binary stream and extracts the header data (everything up to and including the SOS header)
    and the compressed pixel data (entropy-coded DCT coefficients after SOS header up to but not including EOI).
    
    Args:
        jpg_bytes (bytes): The binary data of the JPEG file.
    
    Returns:
        tuple[bytes, bytes]: (header_data, compressed_data)
    
    Raises:
        ValueError: If the file is not a valid JPEG or is malformed.
    """
    if not jpg_bytes.startswith(b'\xff\xd8'):
        raise ValueError("Not a valid JPEG file")
    
    header = bytearray(b'\xff\xd8')
    i = 2
    jpg_length = len(jpg_bytes)
    
    while i < jpg_length:
        if jpg_bytes[i] != 0xff:
            raise ValueError(f"Expected marker at position {i}")
        
        marker = jpg_bytes[i + 1]
        print(f"[jpg_parse] Marker at {i}: FF {marker:02X}")
        header.extend(jpg_bytes[i:i+2])
        i += 2
        
        if marker == 0xd9:  # EOI
            raise ValueError("EOI encountered before SOS")
        
        if marker == 0xda:  # SOS
            if i + 2 > jpg_length:
                raise ValueError("Truncated length for SOS")
            length = int.from_bytes(jpg_bytes[i:i+2], 'big')
            print(f"[jpg_parse] SOS length: {length}")
            if length < 2:
                raise ValueError("Invalid length for SOS")
            if i + length > jpg_length:
                raise ValueError("Truncated SOS header")
            header.extend(jpg_bytes[i:i + length])
            i += length
            print(f"[jpg_parse] Header end: {bytes(header[-10:]).hex()}")
            break
        
        elif marker in range(0xd0, 0xd8) or marker == 0x01:
            continue
        
        else:
            if i + 2 > jpg_length:
                raise ValueError("Truncated length")
            length = int.from_bytes(jpg_bytes[i:i+2], 'big')
            print(f"[jpg_parse] Marker FF {marker:02X} length: {length}")
            if length < 2:
                raise ValueError("Invalid length")
            if i + length > jpg_length:
                raise ValueError("Truncated segment")
            header.extend(jpg_bytes[i:i + length])
            i += length
    
    # Extract compressed data up to EOI, handling byte stuffing and restart markers
    compressed = bytearray()
    while i < jpg_length - 1:
        if jpg_bytes[i] == 0xff:
            next_byte = jpg_bytes[i + 1]
            if next_byte == 0xd9:  # EOI
                break
            compressed.extend(jpg_bytes[i:i+2])  # Include FF 00 or FF D0-D7
            i += 2
        else:
            compressed.append(jpg_bytes[i])
            i += 1
    
    if i >= jpg_length:
        raise ValueError("No EOI found")
    
    print(f"[jpg_parse] Compressed data length: {len(compressed)} bytes")
    print(f"[jpg_parse] First 10 compressed bytes: {bytes(compressed[:10]).hex()}")
    
    if i + 2 < jpg_length:
        print("Warning: There is data after EOI")
    
    return bytes(header), bytes(compressed)

def jpg_build(header: bytes, compressed: bytes) -> bytes:
    """
    Builds a valid JPEG binary stream from the header data and compressed pixel data.
    
    Args:
        header (bytes): The header data (up to and including SOS header).
        compressed (bytes): The compressed pixel data (entropy-coded DCT coefficients).
    
    Returns:
        bytes: The complete JPEG binary stream.
    """
    return header + compressed + b'\xff\xd9'


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


# Path to the input image (replace with your actual image path)
PATH_TO_VIDEO = "/home/pi/Documents/matan/code/D2A2D/1572378-sd_960_540_24fps.mp4"
FRAME_WIDTH = 720
FRAME_HEIGHT = 480

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

    # Read the image
    img = frame_proc.copy()

    # Encode to JPEG binary using OpenCV
    start_time = time.time()
    params = [(cv2.IMWRITE_JPEG_QUALITY), 30, int(cv2.IMWRITE_JPEG_RST_INTERVAL), 1]  # Adjust interval; smaller = more resilient but larger file
    success, jpg_bytes = cv2.imencode('.jpg', img, params)
    if not success:
        raise ValueError("Failed to encode image")

    # Parse
    start_parse = time.time()
    header, compressed = jpg_parse(jpg_bytes.tobytes())
    print(f"Parsed JPEG in {time.time() - start_parse} seconds")
    noisy = np.frombuffer(compressed, dtype=np.uint8).copy()

    noise_mask = np.random.choice([0, 255], size=noisy.shape,
                                                p=[0.99, 0.01]).astype(np.uint8) # 3% bit flips
    noisy ^= noise_mask

    start_fix = time.time()
    fixed_compressed = fix_false_markers(noisy.tobytes())
    print(f"Fixed false markers in {time.time() - start_fix} seconds")

    # if fixes:
    #     print("Fixed false markers:", fixes)

    # Build back
    start_rebuild = time.time()
    rebuilt_bytes = jpg_build(header, fixed_compressed)
    print(f"Rebuilt JPEG in {time.time() - start_rebuild} seconds")

    # Decode back with OpenCV to verify
    rebuilt_img = cv2.imdecode(np.frombuffer(rebuilt_bytes, np.uint8), cv2.IMREAD_COLOR)
    print(f"Total processing time: {time.time() - start_time} seconds")
    if rebuilt_img is None:
        raise ValueError("Failed to decode rebuilt image")
    print("Rebuilt image shape:", rebuilt_img.shape)  # Should match original image shape

    # Save the rebuilt image to verify visually
    cv2.imwrite("rebuilt_image.jpg", rebuilt_img)
    # print("Rebuilt image saved as 'rebuilt_image.jpg'")