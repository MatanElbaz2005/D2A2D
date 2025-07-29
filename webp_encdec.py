import numpy as np
from PIL import Image
import io

# Config (same as your spread-spectrum setup for reference)
FRAME_WIDTH = 720
FRAME_HEIGHT = 480  # NTSC; use 576 for PAL

def encode_frame_to_webp(frame_np: np.ndarray, quality: int = 75) -> bytes:
    """
    Encode a numpy frame to WebP bytes with low-latency, error-resilient settings.
    
    Args:
        frame_np: numpy.ndarray (H, W, 3) RGB or (H, W) grayscale, uint8.
        quality: WebP quality (0-100, higher = better quality, less compression).
    
    Returns:
        bytes: WebP-encoded image data, suitable for spread-spectrum encoding.
    """
    # Convert numpy array to PIL Image
    if len(frame_np.shape) == 2:  # Grayscale
        frame_np = np.stack([frame_np] * 3, axis=-1)  # Convert to RGB for WebP
    img = Image.fromarray(frame_np, mode='RGB')
    
    # Encode to WebP with lossy compression, optimized for speed and resilience
    buffer = io.BytesIO()
    img.save(buffer, format='WEBP', quality=quality, method=6)  # method=6 for fastest encoding
    return buffer.getvalue()

def decode_webp_to_frame(webp_bytes: bytes) -> np.ndarray:
    """
    Decode WebP bytes back to a numpy frame, handling errors gracefully.
    
    Args:
        webp_bytes: bytes, WebP-encoded image data (possibly noisy).
    
    Returns:
        numpy.ndarray: Decoded frame (H, W, 3) RGB, uint8. Returns fallback on error.
    """
    try:
        buffer = io.BytesIO(webp_bytes)
        img = Image.open(buffer, formats=['WEBP'])
        return np.array(img)
    except Exception as e:  # Handle corrupted WebP (e.g., bit errors from VTX noise)
        print(f"WebP decode error: {e}. Returning fallback frame.")
        return np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)  # Black frame
