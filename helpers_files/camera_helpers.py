import cv2
import time
from helpers_files.config_helpers import _cfg

def open_capture():
    """
    OpenCV capture opener that supports file or camera with OS-specific backends and fallbacks.
    Returns (cap, fps).
    """
    cfg = _cfg()
    input_source = cfg["camera"]["input_source"].lower()
    os_name     = cfg["runtime"]["os"].lower()
    path        = cfg["camera"]["path_to_video"]
    cam_index   = cfg["camera"]["camera_index"]
    width       = cfg["frame"]["width"]
    height      = cfg["frame"]["height"]
    target_fps  = float(cfg["camera"]["target_fps"])

    # Case 1: Video file
    if input_source == "file":
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video file: {path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        return cap, float(fps)

    # Case 2: Live camera
    backends = []

    if os_name == "windows":
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]

    elif os_name in ("raspberry_pi", "raspberry"):
        gst_bgr = (
            "libcamerasrc ! "
            "video/x-raw,format=NV12,width=1536,height=864,framerate=30/1 ! "
            "videoconvert ! "
            'video/x-raw,format=BGR ! '
            "appsink name=appsink drop=true max-buffers=1 sync=false caps=video/x-raw,format=BGR"
        )
        gst_bgrx = (
            "libcamerasrc ! "
            "video/x-raw,format=NV12,width=1536,height=864,framerate=30/1 ! "
            "videoconvert ! "
            'video/x-raw,format=BGRx ! '
            "appsink name=appsink drop=true max-buffers=1 sync=false caps=video/x-raw,format=BGRx"
        )

        # BGR
        cap = cv2.VideoCapture(gst_bgr, cv2.CAP_GSTREAMER)
        if not cap.isOpened():
            raise RuntimeError("Failed to open CSI camera via GStreamer/libcamera. "
                            "Check that OpenCV was built with GStreamer and libcamera is installed.")

        # warm-up
        t0 = time.time()
        ok = False; frame = None
        while time.time() - t0 < 3.0:
            ok, frame = cap.read()
            if ok and frame is not None and frame.size:
                try:
                    hh, ww = frame.shape[:2]
                    print(f"[Gst/OpenCV] First frame via BGR: {ww}x{hh}, dtype={frame.dtype}")
                except Exception:
                    pass
                break
            time.sleep(0.01)

        # fallback to BGRx
        if not ok:
            cap.release()
            cap = cv2.VideoCapture(gst_bgrx, cv2.CAP_GSTREAMER)
            if not cap.isOpened():
                raise RuntimeError("CSI camera open failed (BGR and BGRx). Check OpenCV+GStreamer build.")
            t0 = time.time(); ok = False; frame = None
            while time.time() - t0 < 3.0:
                ok, frame = cap.read()
                if ok and frame is not None and frame.size:
                    try:
                        hh, ww = frame.shape[:2]
                        print(f"[Gst/OpenCV] First frame via BGRx: {ww}x{hh}, dtype={frame.dtype}")
                    except Exception:
                        pass
                    break
                time.sleep(0.01)

        if not ok:
            cap.release()
            raise RuntimeError("Camera opened but no frames arrived after BGR and BGRx trials (caps negotiation failed).")

        # libcamera לא תמיד מדווח FPS אמין ל-OpenCV; נחזיר את היעד
        return cap, float(target_fps)

    else:
        backends = [cv2.CAP_ANY]

    last_err = None
    for be in backends:
        try:
            cap = cv2.VideoCapture(cam_index, be)
            if not cap.isOpened():
                cap.release()
                cap = cv2.VideoCapture(cam_index)
            if not cap.isOpened():
                continue

            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS,         target_fps)

            if os_name == "windows":
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

            # warm-up
            for _ in range(8):
                cap.read()
                time.sleep(0.01)

            fps = cap.get(cv2.CAP_PROP_FPS)
            if not fps or fps < 1:
                fps = target_fps

            return cap, float(fps)
        except Exception as e:
            last_err = e
            try:
                cap.release()
            except:
                pass
            continue

    raise RuntimeError(f"Could not open camera index {cam_index}. Last error: {last_err}")
