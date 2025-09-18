import sys, os, base64
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import cv2, numpy as np
from helpers_files.config_helpers import _cfg
from helpers_files.camera_helpers import open_capture
from key_files.decode import decode_frame_to_udp

cfg = _cfg()
W = cfg["frame"]["width"]; H = cfg["frame"]["height"]

# headers from config
hdr_b64 = cfg.get("jpeg", {}).get("header_template_b64", "")
if not hdr_b64:
    raise RuntimeError("jpeg.header_template_b64 missing in config.yaml. "
                       "Run your headers generator step to populate it.")
HEADER_TEMPLATE = base64.b64decode(hdr_b64)
HEADER_TEMPLATE_READY = True

# TCP client (connect to sender)
cap, cam_fps = open_capture()

cv2.setUseOptimized(True)
cv2.setNumThreads(0)

cv2.namedWindow("Receiver (Recovered)", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Receiver (Recovered)", W, H)
BLACK = np.zeros((H, W, 3), dtype=np.uint8)

try:
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            cv2.imshow("Receiver (Recovered)", BLACK)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        # derive frame shape from camera
        h, w = frame.shape[:2]
        ch = 1 if frame.ndim == 2 else frame.shape[2]

        try:
            if ch == 1:
                chips = frame  # already gray
            elif ch == 3:
                bgr = frame
                chips = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            else:
                cv2.imshow("Receiver (Recovered)", BLACK)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue
        except Exception:
            cv2.imshow("Receiver (Recovered)", BLACK)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue


        # decode back to JPEG bytes, then to BGR
        try:
            decoded_data = decode_frame_to_udp(chips, None, HEADER_TEMPLATE_READY, HEADER_TEMPLATE)
            img = cv2.imdecode(np.frombuffer(decoded_data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                img = BLACK
            elif img.shape[0] != H or img.shape[1] != W:
                img = cv2.resize(img, (W, H))
        except Exception:
            img = BLACK

        cv2.imshow("Receiver (Recovered)", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()

