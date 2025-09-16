import cv2, time, numpy as np
from helpers_files.camera_helpers import open_capture
from protected_jpeg import split_jpeg
from key_files.encode import encode_udp_to_frame_dataonly
from key_files.decode import decode_frame_to_udp
from helpers_files.config_helpers import _cfg

cfg = _cfg()
W = cfg["frame"]["width"]
H = cfg["frame"]["height"]

cv2.setUseOptimized(True)
cv2.setNumThreads(0)

cap, cam_fps = open_capture()

cv2.namedWindow("Recovered", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Recovered", W, H)

HEADER_TEMPLATE = None
HEADER_TEMPLATE_READY = False

prev_show_ms = 0.0
while cap.isOpened():
    t0 = time.time()
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
    
    noisy = cv2.cvtColor(enc_frame, cv2.COLOR_GRAY2BGR)
    # noisy += np.random.normal(0.0, sigma, noisy.shape).astype(np.float32)
    # noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    noisy_gray = cv2.cvtColor(noisy, cv2.COLOR_BGR2GRAY)

    decoded_data = decode_frame_to_udp(noisy_gray, None, HEADER_TEMPLATE_READY, HEADER_TEMPLATE)
    decoded_np = np.frombuffer(decoded_data, dtype=np.uint8)
    rec = cv2.imdecode(decoded_np, cv2.IMREAD_COLOR)
    if rec is None:
        rec = np.zeros((H, W, 3), dtype=np.uint8)
    elif rec.shape[0] != H or rec.shape[1] != W:
        rec = cv2.resize(rec, (W, H))

    t1 = time.time()
    dt_cap_to_disp_ms = int((t1 - t0) * 1000.0)

    vis = rec.copy()
    cv2.putText(vis, f"cap->disp: {dt_cap_to_disp_ms} ms", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 3, cv2.LINE_AA)
    cv2.putText(vis, f"cap->disp: {dt_cap_to_disp_ms} ms", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 1, cv2.LINE_AA)

    now_ms = time.time() * 1000.0
    frame_time_ms = int(now_ms - prev_show_ms) if prev_show_ms > 0 else 0
    cv2.putText(vis, f"frame_time: {frame_time_ms} ms", (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 3, cv2.LINE_AA)
    cv2.putText(vis, f"frame_time: {frame_time_ms} ms", (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 1, cv2.LINE_AA)
    prev_show_ms = now_ms

    cv2.imshow("Recovered", vis)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
