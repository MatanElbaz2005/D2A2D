import cv2
import time

def open_capture(input_source: str, os_name: str, path: str, cam_index: int, width: int, height: int, target_fps: float):
    """
    OpenCV capture opener that supports file or camera with OS-specific backends and fallbacks.
    Returns (cap, fps).
    """
    input_source = (input_source or "").lower()
    os_name = (os_name or "").lower()

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
        # Try DirectShow (often best), then Media Foundation, then default
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    elif os_name in ("raspberry_pi", "raspberry"):
        gst = (
            "libcamerasrc ! "
            "video/x-raw,width={w},height={h},framerate={fps}/1 ! "
            "videoconvert ! video/x-raw,format=BGR ! "
            "appsink drop=true max-buffers=1 sync=false"
        ).format(w=width, h=height, fps=int(target_fps))
        cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
        if not cap.isOpened():
            raise RuntimeError("Failed to open CSI camera via GStreamer/libcamera. "
                            "Check that OpenCV was built with GStreamer and libcamera is installed.")
        # libcamera/appsink don’t reliably report FPS; just use target
        return cap, float(target_fps)
    else:
        backends = [cv2.CAP_ANY]

    last_err = None
    for be in backends:
        try:
            cap = cv2.VideoCapture(cam_index, be)
            if not cap.isOpened():
                # try plain constructor if backend form failed
                cap.release()
                cap = cv2.VideoCapture(cam_index)
            if not cap.isOpened():
                continue

            # Try to set desired mode
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS,         target_fps)

            if os_name == "windows":
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

            # Warm-up: let exposure/white-balance settle
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
