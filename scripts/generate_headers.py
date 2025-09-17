import base64, cv2, numpy as np, sys, os
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from protected_jpeg import split_jpeg

yaml_rt = YAML()
yaml_rt.preserve_quotes = True
yaml_rt.width = 4096
yaml_rt.indent(mapping=2, sequence=4, offset=2)

def main(cfg_path="config.yaml"):
    if not os.path.isabs(cfg_path):
        cfg_path = os.path.join(PROJECT_ROOT, cfg_path)

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg_raw = yaml_rt.load(f)

    W = int(cfg_raw["frame"]["width"])
    H = int(cfg_raw["frame"]["height"])
    jpeg_block = cfg_raw.get("jpeg") or CommentedMap()
    q   = int(jpeg_block.get("quality", 70))
    rst = int(jpeg_block.get("rst_interval", 10))

    dummy = np.zeros((H, W, 3), dtype=np.uint8)
    ok, enc = cv2.imencode(".jpg", dummy, [
        cv2.IMWRITE_JPEG_QUALITY, q,
        cv2.IMWRITE_JPEG_RST_INTERVAL, rst
    ])
    if not ok:
        raise RuntimeError("imencode failed")

    headers, _ = split_jpeg(enc.tobytes())
    new_b64 = base64.b64encode(headers).decode("ascii")

    if "jpeg" not in cfg_raw or cfg_raw["jpeg"] is None:
        cfg_raw["jpeg"] = jpeg_block
    cfg_raw["jpeg"]["header_template_b64"] = new_b64

    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml_rt.dump(cfg_raw, f)

    print(f"✅ wrote header_template_b64 ({len(new_b64)} chars) to {cfg_path}")

if __name__ == "__main__":
    main()
