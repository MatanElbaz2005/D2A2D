# D2A2D — Digital → Analog → Digital

## 1) Project Purpose

Transmit **digital video** over an **analog** video channel and reconstruct it back to digital on the other side.
Instead of sending the original frame over the analog channel, we send a **bit-per-pixel encoding** where **black = 0** and **white = 1**.
On top of the raw encoding we add **robustness to noise** (sync markers, PRBS spreading, Reed–Solomon ECC) so the receiver can recover the image under realistic analog degradation and **without latency build-up**.

> In short: **Digital → Analog (chips frame) → Digital**.

---

## 2) Architecture Overview

Two detached processes (can run on two machines):

* **Sender**
  Capture (libcamera/GStreamer on RPi, or OS backend on Windows) → JPEG encode → **split JPEG headers from data** → encode the JPEG **data** into a binary “chips” frame (bit per pixel) → **add *color* Gaussian noise** to simulate an analog link → transmit the resulting frame.

* **Receiver**
  Receive the transmitted **color** frame → convert to **GRAY** (as the decoder expects chips in gray) → **decode back to JPEG bytes** using a **precomputed header template** from config → `cv2.imdecode` → display.

Shared modules (helpers, config, protected JPEG, ECC, binxcorr) exist on both sides and must be identical.

---

## 3) Prerequisites & Setup

> Works on **Raspberry Pi** (recommended for camera + libcamera) and **Windows** (desktop testing).
> On Raspberry Pi, **create a virtual environment** *before* installing Python dependencies.

### 3.1 OS & Python

* Raspberry Pi OS (Bookworm or later) / Windows 10–11
* Python **3.10+**
* Create a venv (Pi highly recommended):

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

### 3.2 Python dependencies

```bash
pip install -r requirements.txt
```

#### Use **creedsolo** (C-backed RS) instead of `reedsolo`

```bash
pip uninstall -y reedsolo
pip install creedsolo
# quick sanity:
python - << 'PY'
from creedsolo import RSCodec
print("creedsolo OK:", RSCodec is not None)
PY
```

### 3.3 Build the C++ extension (`binxcorr`)

Accelerated correlation/sync via pybind11:

```bash
pip install pybind11
# Linux/RPi:
c++ -O3 -Wall -shared -std=c++17 -fPIC \
  $(python3 -m pybind11 --includes) \
  binxcorr.cpp -o binxcorr$(python3-config --extension-suffix)

# test:
python - << 'PY'
import binxcorr
print("binxcorr OK")
PY
```

### 3.4 GStreamer & libcamera (low-latency capture)

**Raspberry Pi:**

```bash
sudo apt update
sudo apt install -y libcamera0 libcamera-apps \
  gstreamer1.0-tools gstreamer1.0-libav \
  gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad
```

Ensure your OpenCV build has **GStreamer** support. The capture pipeline uses:

```
appsink enable-last-sample=false drop=true max-buffers=1 sync=false
```

to avoid buffering and latency accumulation.

**Windows:** you can use MSMF/DSHOW; GStreamer is optional for testing.

---

## 4) Configuration (`config.yaml`)

The configuration is **required** and must be identical on both sides.

```yaml
frame:
  width: 720
  height: 480            # target processing size

rs:
  use_for_headers: true  # RS on headers block
  use_for_data: false    # optional RS on payload chunks
  ecc_symbols: 50        # RS parity symbols per block
  chunk_bytes: 150       # RS data block size (bytes)

prbs:
  use_for_headers: true
  use_for_data: false
  chip_length_for_headers: 3   # spreading for sync/headers
  chip_length_for_data: 1      # spreading for payload
  data_prbs_poly: [8, 2]       # PRBS taps (example)

length:
  bits_per_field: 32     # length fields bit-width
  chip_length: 7         # spreading for length block

markers:
  use: true
  codeword_len: 64       # marker codeword length (chips)
  det_thresh: 0.8        # detection threshold

camera:
  input_source: camera   # "camera" or "file"
  camera_index: 0
  target_fps: 25.0       # advisory; libcamera dictates fps on Pi
  path_to_video: "/path/to/video.mp4"

noise:
  gauss_noise: 50.0      # initial sigma for Gaussian noise (sender UI)

runtime:
  save: false
  os: raspberry_pi       # "windows" or "raspberry_pi"
  start_frame: 5
  num_frames: 10

jpeg:
  quality: 70
  rst_interval: 10
  header_template_b64: "<BASE64-OF-JPEG-HEADERS>"   # REQUIRED

transport:
  host: "0.0.0.0"           # sender bind address
  connect_host: "127.0.0.1" # receiver connects here
  port: 5001
```

### About `jpeg.header_template_b64` (REQUIRED)

Sender and Receiver are detached; the **receiver must reconstruct** JPEG bytes using the **exact headers** the sender used (quality, RST, color space, etc.).
We store those headers as Base64 in `jpeg.header_template_b64`.

**Generate the headers once** with the provided script:

```bash
# from repo root
python D2A2D/scripts/generate_headers.py
```

This will bake the header template (Base64) into your `config.yaml`.
Re-run whenever you change size/quality/RST.

---

## 5) Running

There are **three** supported run modes:

### A) Single-process demo — `main.py`

Encode and decode on the **same machine**, simulating ***color* Gaussian noise** between them.

```bash
python main.py
```

### B) Single-process debug demo — `main_with_debug.py`

Same as A, plus **three additional debug views**:

* Original video
* Analog video (how a regular analog feed would look; no digital chips)
* Encoded chips (our black/white bit-per-pixel frame)
* Recovered (decoded result)

```bash
python main_with_debug.py
```

### C) Split processes — **Sender / Receiver**

**Sender** captures → encodes to chips → adds **color** noise → **transmits** (TCP).
**Receiver** receives **color**, converts to **GRAY**, decodes JPEG, and displays.

Run in two terminals:

```bash
python sender/sender_main.py
python receiver/receiver_main.py
```

Make sure:

* `jpeg.header_template_b64` is present (generated as above),
* `frame` / `jpeg` params are identical on both sides,
* `transport` addresses/port are correct.

**What goes on the wire (TCP):**
A tiny header:

```
magic='D2A2'(4), ver(1), flags(1), width(2), height(2), channels(1),
payload_len(4), ts_usec(8)
```

Payload is raw frame bytes (`uint8`), **channels = 3** (BGR).
Receiver converts BGR→GRAY **before** decode, because the decoder expects gray chips.

---

## 6) Protection Pipeline (Encode/Decode)

* **JPEG split/merge**: encode the resized frame to JPEG, **split headers and data**; only **data** becomes chips. Receiver merges recovered data with the **header template** from config → valid JPEG bytes.

* **Markers & Sync**: distinctive **marker codewords** (`markers.codeword_len`) and **Gold-style sync sequences** (±1). The receiver locates sync via **correlation** (accelerated by the `binxcorr` C++ extension) and realigns bit boundaries under noise.

* **Length block**: dedicated, spread by its own `length.chip_length`, carrying the payload size so the receiver knows how many bits/bytes to collect.

* **PRBS spreading**: (as configured) whitening improves robustness against colored noise and helps correlation.

* **ECC (Reed–Solomon)** via **creedsolo**: parity over headers (and optionally data) with `rs.ecc_symbols`/`rs.chunk_bytes`, providing strong burst-error correction.

* **Realtime discipline**:

  * Capture via **appsink** with `drop=true max-buffers=1 sync=false` (no frame queue).
  * Transport keeps only **latest** frame (drop-old), so **no latency accumulation**.
  * We **don’t force FPS**; we follow the camera. If compute is slower, frames get dropped—**latency stays low**.

---

## 7) Troubleshooting

**Receiver: `ConnectionRefusedError`**

* Start the sender first; you should see a “Listening on …” log.
* Verify `transport.connect_host`/`port` match the sender.
* On Windows, prefer `127.0.0.1` (localhost nuances, firewall).

**Missing `jpeg.header_template_b64`**

* Run the headers generator (see §4) after deciding width/height/quality/RST.
* Both sides must use the **same** header template and frame size.

**`creedsolo` not found / build errors\`**

* `pip install creedsolo`. If your platform needs build tools:
  `sudo apt install build-essential python3-dev` (Linux/RPi).
* The code can fall back to `reedsolo`, but **performance is much better with creedsolo**.

**`binxcorr` import error\`**

* Rebuild with §3.3. Ensure the compiled `.so` is in the repo root or on `PYTHONPATH`.
* Test import as shown.

**Camera lag / buffering**

* Ensure your GStreamer pipeline uses `appsink enable-last-sample=false drop=true max-buffers=1 sync=false`.
* Remove any extra queues that accumulate frames.

**Low FPS / CPU bound**

* Reduce `frame.width/height`.
* Lower `jpeg.quality` or increase `jpeg.rst_interval`.
* Disable heavy debug overlays.
* Confirm `cv2.setUseOptimized(True)` and `cv2.setNumThreads(0)`.

**Mismatched sizes or color**

* `frame.width/height` must match both sides.
* Sender transmits **BGR (3 channels)**; Receiver converts to **GRAY** before decode.

---
