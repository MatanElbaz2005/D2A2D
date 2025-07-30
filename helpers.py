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
