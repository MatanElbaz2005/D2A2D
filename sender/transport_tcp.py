import socket, struct, threading, time

# header: magic(4)='D2A2', ver(1), flags(1), width(2), height(2), channels(1), payload_len(4), ts_usec(8)
_HDR_FMT = "<4sBBHHBIQ"
_HDR_SIZE = struct.calcsize(_HDR_FMT)
_MAGIC = b"D2A2"
_VER = 1

def pack_header(width: int, height: int, channels: int, payload_len: int, flags: int = 0):
    ts_usec = int(time.time() * 1_000_000)
    return struct.pack(_HDR_FMT, _MAGIC, _VER, flags, width, height, channels, payload_len, ts_usec)

def unpack_header(b: bytes):
    magic, ver, flags, w, h, ch, plen, tsu = struct.unpack(_HDR_FMT, b)
    if magic != _MAGIC or ver != _VER:
        raise ValueError("Bad header")
    return {"flags": flags, "width": w, "height": h, "channels": ch, "payload_len": plen, "ts_usec": tsu}

def recvall(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Socket closed")
        buf.extend(chunk)
    return bytes(buf)

class TcpServer:
    """Minimal single-client TCP server. Keeps latest frame only (drop old)."""
    def __init__(self, host="0.0.0.0", port=5001):
        self.host, self.port = host, port
        self._srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._srv.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._srv.bind((self.host, self.port))
        self._srv.listen(1)
        self._cli = None
        self._lock = threading.Lock()
        self._latest = None  # last (header, payload) tuple
        self._running = True
        self._th = threading.Thread(target=self._accept_loop, daemon=True)
        self._ths = threading.Thread(target=self._send_loop, daemon=True)
        self._th.start()
        self._ths.start()

    def _accept_loop(self):
        while self._running:
            try:
                cli, _ = self._srv.accept()
                cli.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                with self._lock:
                    if self._cli:
                        try: self._cli.close()
                        except: pass
                    self._cli = cli
            except Exception:
                time.sleep(0.05)

    def publish_gray(self, width: int, height: int, payload: bytes, channels: int = 1):
        hdr = pack_header(width, height, channels, len(payload))
        with self._lock:
            self._latest = (hdr, payload)

    def _send_loop(self):
        while self._running:
            hdr_payload = None
            with self._lock:
                if self._latest is not None:
                    hdr_payload = self._latest
                    self._latest = None  # drop old; keep only newest
            if hdr_payload and self._cli:
                try:
                    self._cli.sendall(hdr_payload[0])
                    self._cli.sendall(hdr_payload[1])
                except Exception:
                    try: self._cli.close()
                    except: pass
                    self._cli = None
            else:
                time.sleep(0.001)

    def close(self):
        self._running = False
        try: self._srv.close()
        except: pass
        try:
            with self._lock:
                if self._cli: self._cli.close()
        except: pass


class TcpClient:
    """Blocking receive API: recv_frame() -> (ok, header_dict, payload_bytes)"""
    def __init__(self, host="127.0.0.1", port=5001, timeout=3.0):
        self.host, self.port = host, port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock.settimeout(timeout)
        self.sock.connect((self.host, self.port))
        # after connect, optional shorter timeout
        self.sock.settimeout(0.5)

    def recv_frame(self):
        try:
            hdr = recvall(self.sock, _HDR_SIZE)
            h = unpack_header(hdr)
            payload = recvall(self.sock, h["payload_len"])
            return True, h, payload
        except Exception:
            return False, None, None

    def close(self):
        try: self.sock.close()
        except: pass
