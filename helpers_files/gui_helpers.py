import numpy as np
import cv2

def _to_bgr(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img

def _label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.putText(out, text, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(out, text, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 1, cv2.LINE_AA)
    return out

def _compose_grid(tl: np.ndarray, tr: np.ndarray, bl: np.ndarray, br: np.ndarray, gap: int = 20) -> np.ndarray:
    h, w = tl.shape[:2]
    canvas = np.full((h*2 + gap, w*2 + gap, 3), 24, dtype=np.uint8)
    canvas[0:h, 0:w] = tl
    canvas[0:h, w+gap:w*2+gap] = tr
    canvas[h+gap:h*2+gap, 0:w] = bl
    canvas[h+gap:h*2+gap, w+gap:w*2+gap] = br
    return canvas
