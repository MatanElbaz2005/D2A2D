import yaml
from functools import lru_cache
try:
    from creedsolo import RSCodec, ReedSolomonError
except ImportError:
    print("cant find creedsolo, using reedsolo instead")
    from reedsolo import RSCodec, ReedSolomonError
import numpy as np

@lru_cache(maxsize=1)
def _cfg():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

@lru_cache(maxsize=1)
def get_rsc():
    return RSCodec(_cfg()["rs"]["ecc_symbols"])

@lru_cache(maxsize=None)
def get_prbs(length: int, poly: tuple[int, ...], seed: int = 3):
    from helpers_files.helpers import generate_prbs
    return generate_prbs(length, list(poly), seed)

@lru_cache(maxsize=1)
def get_headers_sync():
    from helpers_files.helpers import gold127
    return gold127(shift=0).astype(np.int8, copy=False)

@lru_cache(maxsize=1)
def get_marker_codebook():
    cfg = _cfg()
    L = cfg["markers"]["codeword_len"]
    MARKER_TOKENS = [bytes([0xFF, 0xD0 + i]) for i in range(8)] + [b"\xFF\x00"]
    from helpers_files.helpers import _build_marker_codewords_gold
    TOKENS, CODES = _build_marker_codewords_gold(L, MARKER_TOKENS)
    return TOKENS, CODES

