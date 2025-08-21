import numpy as np

_PM_LUT8 = (np.unpackbits(np.arange(256, dtype=np.uint8).reshape(-1,1), axis=1).astype(np.int8)*2 - 1)  # shape (256,8)
POPCNT8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)

def conv_encode_bits(bits: np.ndarray, g1: int = 0o133, g2: int = 0o171) -> np.ndarray:
    k = 7
    state = 0
    out = []
    for b in bits:
        state = ((state << 1) | int(b)) & ((1 << k) - 1)
        o1 = bin(state & g1).count("1") & 1
        o2 = bin(state & g2).count("1") & 1
        out.extend([o1, o2])
    return np.array(out, dtype=np.uint8)

def viterbi_decode_bits(
    coded: np.ndarray, tb_depth: int = 15, g1: int = 0o133, g2: int = 0o171
) -> np.ndarray:
    k = 7
    n_states = 1 << (k - 1)
    # pre‑compute branch outputs for speed
    branch = np.zeros((n_states, 2, 2), dtype=np.uint8)  # [state][bit][parity‑idx]
    for s in range(n_states):
        for bit in (0, 1):
            nxt = ((s << 1) | bit) & (n_states - 1)
            o1 = bin(((s << 1) | bit) & g1).count("1") & 1
            o2 = bin(((s << 1) | bit) & g2).count("1") & 1
            branch[s, bit] = (o1, o2)
    # initialise metrics
    inf = 1_000_000
    path_metric = np.full(n_states, inf, dtype=np.int32)
    path_metric[0] = 0
    prev_state = np.zeros((len(coded) // 2, n_states), dtype=np.uint8)
    # forward recursion
    for t in range(0, len(coded), 2):
        sym = coded[t : t + 2]
        pm_next = np.full(n_states, inf, dtype=np.int32)
        for s in range(n_states):
            for bit in (0, 1):
                nxt = ((s << 1) | bit) & (n_states - 1)
                metric = path_metric[s] + np.sum(branch[s, bit] ^ sym)
                if metric < pm_next[nxt]:
                    pm_next[nxt] = metric
                    prev_state[t // 2, nxt] = (s << 1 | bit) & 0xFF
        path_metric = pm_next
    # trace‑back
    state = np.argmin(path_metric)
    decoded = []
    for t in range(len(prev_state) - 1, -1, -1):
        bit = state & 1
        decoded.append(bit)
        state = prev_state[t, state] >> 1
        if len(decoded) >= tb_depth:
            pass  # keep tracing
    return np.array(decoded[::-1], dtype=np.uint8)

def generate_prbs(length: int, poly: list[int], seed: int | None = None) -> np.ndarray:
    """
    Generate a ±1 pseudo-random binary sequence (PRBS) with a linear-feedback shift register.

    Parameters
    ----------
    length : int
        Number of output chips to produce.
    poly   : list[int]
        Tap polynomial for the LFSR.  
        `poly[0]` is the register size N; the remaining values are the tap
        positions (counted from MSB, 1-based).  
        Example ``[6, 5]`` → 6-bit LFSR with feedback taps at bits 6 and 5.
    seed   : int, optional
        Initial register state.  If *None*, the register is initialised to
        all ones.  A zero state is never allowed because it would lock the LFSR.

    Returns
    -------
    numpy.ndarray of shape ``(length,)`` and dtype ``int8``
        The PRBS expressed as +1 / −1 chips.
    """
    degree = poly[0]                                 # register size (bits)
    if seed is None:
        state = (1 << degree) - 1                    # default seed: 0b111…1
    else:
        state = seed & ((1 << degree) - 1)           # mask to degree bits

    out  = np.empty(length, dtype=np.int8)
    taps = [degree - t for t in poly[1:]]            # shifts for XOR taps

    for i in range(length):
        lsb        = state & 1                       # output bit (LSB)
        out[i]     = 1 if lsb else -1                # map {0,1} → {-1,+1}
        feedback   = 0
        for sh in taps:                              # XOR of tap bits
            feedback ^= (state >> sh) & 1
        state = (state >> 1) | (feedback << (degree - 1))

    return out

def block_interleave(data: bytes, rows: int) -> bytes:
    cols = int(np.ceil(len(data) / rows))
    padded = data + bytes(cols * rows - len(data))
    matrix = np.frombuffer(padded, dtype=np.uint8).reshape(rows, cols)
    return bytes(matrix.T.flatten()[:len(data)])

def block_deinterleave(data: bytes, rows: int) -> bytes:
    cols = int(np.ceil(len(data) / rows))
    padded = data + bytes(cols * rows - len(data))
    matrix = np.frombuffer(padded, dtype=np.uint8).reshape(cols, rows).T
    return bytes(matrix.flatten()[:len(data)])

def _mseq_127_taps_7_3(seed=0x7F):
    state = seed & 0x7F
    out = np.empty(127, np.uint8)
    for i in range(127):
        out[i] = state & 1
        fb = ((state >> 6) ^ (state >> 2)) & 1
        state = ((state >> 1) | (fb << 6)) & 0x7F
    return out

def _mseq_127_taps_7_1(seed=0x7F):
    state = seed & 0x7F
    out = np.empty(127, np.uint8)
    for i in range(127):
        out[i] = state & 1
        fb = ((state >> 6) ^ (state >> 0)) & 1
        state = ((state >> 1) | (fb << 6)) & 0x7F
    return out

def gold127(shift=0):
    m1 = _mseq_127_taps_7_3()
    m2 = _mseq_127_taps_7_1()
    g = np.bitwise_xor(m1, np.roll(m2, shift))
    return (g.astype(np.int32) * 2 - 1)


def _is_marker_token_at(data: bytes, i: int) -> bool:
    return (i + 1 < len(data)) and data[i] == 0xFF and (data[i+1] == 0x00 or 0xD0 <= data[i+1] <= 0xD7)

def _best_balanced_segment_pm(seq_pm: np.ndarray, L: int) -> np.ndarray:
    """Pick a length-L circular segment of a ±1 sequence with minimal |sum| (≈ mean 0)."""
    s = seq_pm.astype(np.int8)
    n = s.size
    x = np.concatenate([s, s])  # allow wraparound
    c = np.concatenate(([0], np.cumsum(x, dtype=np.int32)))
    best = (10**9, 0)
    for start in range(n):
        end = start + L
        seg_sum = int(c[end] - c[start])  # sum of ±1
        score = abs(seg_sum)
        if score < best[0]:
            best = (score, start)
            if score == 0:
                break
    start = best[1]
    return x[start:start+L].astype(np.int8)

def _build_marker_codewords_gold(L: int, MARKER_TOKENS) -> tuple[list[bytes], np.ndarray]:
    """
    Build ±1 codewords from Gold(127) with different shifts, then pick a balanced length-L window.
    """
    shifts = [0, 9, 17, 29, 37, 45, 53, 61, 73]  # well-separated
    codes = []
    for sh in shifts:
        g = gold127(shift=sh).astype(np.int8)  # ±1 length 127
        cw = _best_balanced_segment_pm(g, L)   # choose balanced window of length L
        codes.append(cw)
    C = np.stack(codes, axis=0).astype(np.int8)
    return MARKER_TOKENS, C



def _encode_data_with_codewords(data: bytes, tokens: list[bytes], codes: np.ndarray) -> np.ndarray:
    """
    For each marker token -> emit its codeword (L chips).
    For any other byte -> emit 8 chips (±1 per bit).
    """
    tok2row = {tok: i for i, tok in enumerate(tokens)}
    out = []
    i = 0
    n = len(data)
    while i < n:
        if _is_marker_token_at(data, i):
            tok = data[i:i+2]
            row = tok2row.get(tok, None)
            if row is not None:
                out.append(codes[row])  # L chips
                i += 2
                continue
            # fallback if somehow unknown (shouldn't happen)
        # normal byte
        b = data[i]
        bits = np.unpackbits(np.frombuffer(bytes([b]), np.uint8)).astype(np.int8)*2-1
        out.append(bits)  # 8 chips
        i += 1
    return np.concatenate(out).astype(np.int8)

def _decode_data_with_codewords(chips_pm: np.ndarray, tokens: list[bytes], codes: np.ndarray, thresh: float) -> bytes:
    """
    Greedy sliding-window parser:
      - If corr(window_L, any_codeword) >= thresh -> emit its token, advance L.
      - Else -> read next 8 chips as a normal byte, advance 8.
    """
    L = codes.shape[1]
    out = bytearray()
    pos = 0
    N = chips_pm.size

    def _decode_normal_byte_at(p: int) -> tuple[int, int]:
        byte_chips = chips_pm[p:p+8]
        bits = (byte_chips > 0).astype(np.uint8)
        return int(np.packbits(bits)[0]), p + 8

    while pos < N:
        # try marker match
        if pos + L <= N:
            win = chips_pm[pos:pos+L].astype(np.int32)
            # correlation per codeword (normalized)
            scores = (codes @ win) / L
            j = int(np.argmax(scores))
            if float(scores[j]) >= thresh:
                out.extend(tokens[j])
                pos += L
                continue
        # normal byte
        if pos + 8 > N:
            break
        b, pos = _decode_normal_byte_at(pos)
        out.append(b)

    return bytes(out)

def _encode_data_with_codewords_fast(data: bytes, tokens: list[bytes], codes: np.ndarray) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    n = arr.size
    if n == 0:
        return np.empty(0, dtype=np.int8)

    allowed = np.array([0x00] + list(range(0xD0, 0xD8)), dtype=np.uint8)
    starts_mask = (arr[:-1] == 0xFF) & np.isin(arr[1:], allowed)
    starts = np.flatnonzero(starts_mask)

    second_to_row = np.full(256, -1, dtype=np.int16)
    for i, tok in enumerate(tokens):
        second_to_row[tok[1]] = i

    L = int(codes.shape[1])
    m = int(starts.size)
    out_len = 8*(n - 2*m) + L*m
    out = np.empty(out_len, dtype=np.int8)

    w = 0
    cursor = 0
    for s in starts:
        seg_len = s - cursor
        if seg_len > 0:
            blk = _PM_LUT8[arr[cursor:s]].reshape(-1)
            out[w:w+8*seg_len] = blk
            w += 8*seg_len
        row = int(second_to_row[arr[s+1]])
        if row >= 0:
            out[w:w+L] = codes[row]
            w += L
            cursor = s + 2
        else:
            blk = _PM_LUT8[arr[s:s+1]].reshape(-1)
            out[w:w+8] = blk
            w += 8
            cursor = s + 1

    if cursor < n:
        seg_len = n - cursor
        blk = _PM_LUT8[arr[cursor:]].reshape(-1)
        out[w:w+8*seg_len] = blk
        w += 8*seg_len

    return out

def _decode_data_with_codewords_fast(chips_pm: np.ndarray, tokens: list[bytes], codes: np.ndarray, thresh: float) -> bytes:
    chips_pm = chips_pm.astype(np.int8, copy=False)
    L = int(codes.shape[1])
    N = int(chips_pm.size)
    if N < 8:
        return b""

    step = 8
    n_pos = (N - L) // step + 1 if N >= L else 0

    if n_pos > 0:
        strided = np.lib.stride_tricks.as_strided(
            chips_pm,
            shape=(n_pos, L),
            strides=(chips_pm.strides[0]*step, chips_pm.strides[0])
        )
        scores = (strided.astype(np.int16) @ codes.T.astype(np.int16)) / L
        best_idx = np.argmax(scores, axis=1)
        best_val = scores[np.arange(n_pos), best_idx]
        hits = best_val >= thresh
    else:
        hits = np.zeros(0, dtype=bool)
        best_idx = np.zeros(0, dtype=np.intp)

    out = bytearray()
    pos = 0
    pos8 = 0

    while pos < N:
        if pos8 < hits.size and hits[pos8]:
            j = int(best_idx[pos8])
            out.extend(tokens[j])
            pos  += L
            pos8 += L // 8
        else:
            if pos8 < hits.size:
                rel = hits[pos8:]
                next_rel = np.flatnonzero(rel)
                next_hit = (pos8 + int(next_rel[0])) if next_rel.size else hits.size
            else:
                next_hit = pos8

            run_pos8 = max(0, next_hit - pos8)
            run_chips = chips_pm[pos:pos + 8*run_pos8]
            if run_chips.size > 0:
                bits = (run_chips > 0).astype(np.uint8).reshape(-1, 8)
                out.extend(np.packbits(bits, axis=1).ravel().tolist())
            pos  += 8*run_pos8
            pos8 += run_pos8

            if pos8 >= hits.size:
                rem = N - pos
                tail_bytes = rem // 8
                if tail_bytes > 0:
                    bits = (chips_pm[pos:pos + 8*tail_bytes] > 0).astype(np.uint8).reshape(-1, 8)
                    out.extend(np.packbits(bits, axis=1).ravel().tolist())
                    pos += 8*tail_bytes
                break

    return bytes(out)

def _decode_data_with_codewords_popcnt(chips_pm: np.ndarray, tokens: list[bytes], codes_packed: np.ndarray, L: int, thresh: float) -> bytes:
    chips_pm = chips_pm.astype(np.int8, copy=False)
    N = int(chips_pm.size)
    if N < 8: return b""

    step = 8
    n_pos = (N - L) // step + 1 if N >= L else 0

    if n_pos > 0:
        strided = np.lib.stride_tricks.as_strided(
            chips_pm,
            shape=(n_pos, L),
            strides=(chips_pm.strides[0]*step, chips_pm.strides[0])
        )
        bits = (strided > 0).astype(np.uint8)
        win_packed = np.packbits(bits, axis=1)

        xor = np.bitwise_xor(win_packed[:, None, :], codes_packed[None, :, :]).astype(np.uint8)
        dists = POPCNT8[xor].sum(axis=2) 
        best_idx = np.argmin(dists, axis=1) 
        best_dist = dists[np.arange(n_pos), best_idx]

        d_max = int(np.floor((1.0 - thresh) * L / 2.0))
        hits = (best_dist <= d_max)
    else:
        hits = np.zeros(0, dtype=bool)
        best_idx = np.zeros(0, dtype=np.intp)

    out = bytearray()
    pos = 0
    pos8 = 0
    while pos < N:
        if pos8 < hits.size and hits[pos8]:
            j = int(best_idx[pos8])
            out.extend(tokens[j])
            pos  += L
            pos8 += L // 8
        else:
            if pos8 < hits.size:
                rel = hits[pos8:]
                nz  = np.flatnonzero(rel)
                next_hit = pos8 + int(nz[0]) if nz.size else hits.size
            else:
                next_hit = pos8

            run_b = max(0, next_hit - pos8)
            run_chips = chips_pm[pos:pos + 8*run_b]
            if run_chips.size > 0:
                b = (run_chips > 0).astype(np.uint8).reshape(-1, 8)
                out.extend(np.packbits(b, axis=1).ravel().tolist())
            pos  += 8*run_b
            pos8 += run_b

            if pos8 >= hits.size:
                rem = N - pos
                tail_b = rem // 8
                if tail_b > 0:
                    b = (chips_pm[pos:pos + 8*tail_b] > 0).astype(np.uint8).reshape(-1, 8)
                    out.extend(np.packbits(b, axis=1).ravel().tolist())
                    pos += 8*tail_b
                break

    return bytes(out)
