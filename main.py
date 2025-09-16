import cv2, time, numpy as np
from helpers_files.camera_helpers import open_capture
from protected_jpeg import split_jpeg
from key_files.encode import encode_udp_to_frame_dataonly
from key_files.decode import decode_frame_to_udp
from helpers_files.config_helpers import _cfg

cfg = _cfg()
W = cfg["frame"]["width"]
H = cfg["frame"]["height"]
GAUSS_NOISE = cfg["noise"]["gauss_noise"]

cv2.setUseOptimized(True)
cv2.setNumThreads(0)

cap, cam_fps = open_capture()

cv2.namedWindow("Monitor", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Monitor", W, H)
cv2.createTrackbar("Noise", "Monitor", int(GAUSS_NOISE), 100, lambda v: None)
cv2.setTrackbarPos("Noise", "Monitor", int(GAUSS_NOISE))  

HEADER_TEMPLATE = None
HEADER_TEMPLATE_READY = False

while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        break

    frame_proc = cv2.resize(frame, (W, H), interpolation=cv2.INTER_NEAREST)
    _, encoded = cv2.imencode(".jpg", frame_proc, [cv2.IMWRITE_JPEG_QUALITY, 70, cv2.IMWRITE_JPEG_RST_INTERVAL, 10])
    headers, compressed = split_jpeg(encoded.tobytes())

    if not HEADER_TEMPLATE_READY:
        HEADER_TEMPLATE = headers
        HEADER_TEMPLATE_READY = True
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    enc_frame, tx_meta = encode_udp_to_frame_dataonly(compressed)
    sigma = float(cv2.getTrackbarPos('Noise', 'Monitor'))
    
    noisy = cv2.cvtColor(enc_frame, cv2.COLOR_GRAY2BGR).astype(np.float32)
    noisy += np.random.normal(0.0, sigma, noisy.shape).astype(np.float32)
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    noisy_gray = cv2.cvtColor(noisy, cv2.COLOR_BGR2GRAY)

    decoded_data = decode_frame_to_udp(noisy_gray, None, HEADER_TEMPLATE_READY, HEADER_TEMPLATE)
    decoded_np = np.frombuffer(decoded_data, dtype=np.uint8)
    rec = cv2.imdecode(decoded_np, cv2.IMREAD_COLOR)
    if rec is None:
        rec = np.zeros((H, W, 3), dtype=np.uint8)
    elif rec.shape[0] != H or rec.shape[1] != W:
        rec = cv2.resize(rec, (W, H))

    cv2.imshow("Monitor", rec)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
