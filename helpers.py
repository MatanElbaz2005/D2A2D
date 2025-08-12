import numpy as np

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
