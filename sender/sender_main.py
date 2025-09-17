import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import cv2, numpy as np
from helpers_files.camera_helpers import open_capture
from helpers_files.config_helpers import _cfg
from protected_jpeg import split_jpeg
from key_files.encode import encode_udp_to_frame_dataonly
from sender.transport_tcp import TcpServer

cfg = _cfg()
W = cfg["frame"]["width"]; H = cfg["frame"]["height"]
q = cfg.get("jpeg", {}).get("quality", 40)
rst = cfg.get("jpeg", {}).get("rst_interval", 10)
sigma0 = int(cfg.get("noise", {}).get("gauss_noise", 0))

# TCP server
HOST = cfg.get("transport", {}).get("host", "0.0.0.0")
PORT = int(cfg.get("transport", {}).get("port", 5001))
server = TcpServer(HOST, PORT)
print(f"[Sender] TCP listening on {HOST}:{PORT}")

cv2.setUseOptimized(True)
cv2.setNumThreads(0)

try:
    cap, _ = open_capture()
except Exception as e:
    print(f"[Sender] open_capture failed: {e}. Streaming black frames.")
    cap = None

cv2.namedWindow("Sender (TX chips)", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Sender (TX chips)", W, H)
cv2.createTrackbar("Noise", "Sender (TX chips)", sigma0, 100, lambda v: None)

try:
    while True:
        if cap is not None:
            ok, frame = cap.read()
        else:
            ok, frame = True, np.zeros((H, W, 3), dtype=np.uint8)
        if not ok or frame is None:
            server.publish_gray(W, H, bytes(np.zeros((H, W), np.uint8)))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        frame_proc = cv2.resize(frame, (W, H), interpolation=cv2.INTER_NEAREST)

        ok, enc = cv2.imencode(".jpg", frame_proc,
                               [cv2.IMWRITE_JPEG_QUALITY, q,
                                cv2.IMWRITE_JPEG_RST_INTERVAL, rst])
        if not ok:
            continue

        _headers, data = split_jpeg(enc.tobytes())
        chips_gray, _txmeta = encode_udp_to_frame_dataonly(data, _RUNTIME=None)  # GRAY8

        # simulate analog noise (color then to gray, like in your main)
        sigma = float(cv2.getTrackbarPos("Noise", "Sender (TX chips)"))
        noisy = cv2.cvtColor(chips_gray, cv2.COLOR_GRAY2BGR).astype(np.float32)
        noisy += np.random.normal(0.0, sigma, noisy.shape).astype(np.float32)
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)

        # publish over TCP (drop-old semantics in server)
        server.publish_gray(W, H, noisy.tobytes(), channels=3)

        # local preview (optional)
        cv2.imshow("Sender (TX chips)", noisy)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cap.release()
    server.close()
    cv2.destroyAllWindows()
