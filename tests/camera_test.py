import cv2, time

gst = (
    "libcamerasrc ! "
    "video/x-raw,format=NV12,width=720,height=480,framerate=30/1,colorimetry=bt709 ! "
    "queue max-size-buffers=1 leaky=downstream ! "
    "videoconvert ! "
    "queue max-size-buffers=1 leaky=downstream ! "
    "video/x-raw,format=BGR ! "
    "appsink enable-last-sample=false drop=true max-buffers=1 sync=false"
)


cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
if not cap.isOpened():
    raise RuntimeError("Failed to open CSI camera via GStreamer/libcamera with appsink.")

cv2.namedWindow("Raw", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Raw", 720, 480)

last_show = 0.0
while True:
    t0 = time.time()
    ok, frame = cap.read()
    if not ok:
        break

    dt_ms = int((time.time() - t0) * 1000.0)
    vis = frame
    cv2.putText(vis, f"cap->disp: {dt_ms} ms", (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 3, cv2.LINE_AA)
    cv2.putText(vis, f"cap->disp: {dt_ms} ms", (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 1, cv2.LINE_AA)

    now = time.time()*1000.0
    frame_time_ms = int(now - last_show) if last_show else 0
    last_show = now
    cv2.putText(vis, f"frame_time: {frame_time_ms} ms", (12, 56),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 3, cv2.LINE_AA)
    cv2.putText(vis, f"frame_time: {frame_time_ms} ms", (12, 56),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 1, cv2.LINE_AA)

    cv2.imshow("Raw", vis)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
