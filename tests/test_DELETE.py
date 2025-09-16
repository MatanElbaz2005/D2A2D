import cv2
import numpy as np
from protected_jpeg import split_jpeg

FRAME_WIDTH = 720
FRAME_HEIGHT = 480
ENCODE_PARAM = [
    cv2.IMWRITE_JPEG_QUALITY, 70,
    cv2.IMWRITE_JPEG_RST_INTERVAL, 10
]

dummy_frame = np.full((FRAME_HEIGHT, FRAME_WIDTH, 3), 255, dtype=np.uint8)

success, encoded = cv2.imencode(".jpg", dummy_frame, ENCODE_PARAM)
if not success:
    raise RuntimeError("JPEG encode failed")

jpeg_bytes = encoded.tobytes()

headers, data = split_jpeg(jpeg_bytes)

print(f"Header template size: {len(headers)} bytes")
print("Full header (hex):")
print(headers.hex(" "))

with open("header_template.bin", "wb") as f:
    f.write(headers)
print("Saved header template to 'header_template.bin'")
