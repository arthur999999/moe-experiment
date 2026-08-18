#!/usr/bin/env python3
"""check_cache_purity.py -- prove expert pages are NOT kept resident by streaming.

After a streamer run with DONTNEED, the expert tensors' file pages are dropped
from the page cache: the bytes live only in the LRU/staging buffers.
Non-expert tensors (attention, norms) are read by llama via mmap WITHOUT
DONTNEED, so they should remain resident.

A/B protocol (proves the DONTNEED effect on real data):

    # 1. cold baseline (drops the whole file from page cache)
    python scripts/check_cache_purity.py MODEL sweep

    # 2. stream with DONTNEED ON  (8th arg after cache_mb = 1)
    ./build/phase5_stream MODEL "prompt..." 1024 4 42 0 4 1 1 1 >/dev/null 2>/tmp/run1.log

    # 3. measure
    python scripts/check_cache_purity.py MODEL check

    # 4. repeat 1-2 with DONTNEED OFF (arg = 0), then check again
    python scripts/check_cache_purity.py MODEL check

Expected:
    dontneed=1  -> expert resident LOW    (streamer dropped every expert read)
    dontneed=0  -> expert resident HIGH   (page cache kept everything)
    control     -> HIGH in both           (attn via mmap, no DONTNEED)
"""
import argparse
import ctypes
import mmap
import os
import struct
import sys

# ---------------------------------------------------------------------------
# minimal GGUF parser (stdlib only)
# ---------------------------------------------------------------------------
MAGIC = b"GGUF"
GGUF_TYPE_UINT8, GGUF_TYPE_INT8, GGUF_TYPE_UINT16, GGUF_TYPE_INT16 = 0, 1, 2, 3
GGUF_TYPE_UINT32, GGUF_TYPE_INT32, GGUF_TYPE_FLOAT32, GGUF_TYPE_BOOL = 4, 5, 6, 7
GGUF_TYPE_STRING, GGUF_TYPE_ARRAY = 8, 9
GGUF_TYPE_UINT64, GGUF_TYPE_INT64, GGUF_TYPE_FLOAT64 = 10, 11, 12


class _Reader:
    """Buffered binary reader: reads the file in 4MiB chunks, parses from memory.
    Most GGUF headers are ~10-20MB no matter how big the file is -- this avoids
    one syscall per metadata item (tokenizer arrays have tens of thousands)."""
    CHUNK = 1 << 22  # 4 MiB

    def __init__(self, f):
        self.f = f
        self.buf = b""
        self.pos = 0
        self.abs = 0   # total bytes consumed from the file (for data_offset)

    def _fill(self, n):
        while len(self.buf) - self.pos < n:
            chunk = self.f.read(self.CHUNK)
            if not chunk:
                raise EOFError("unexpected EOF (truncated gguf?)")
            self.buf = self.buf[self.pos:] + chunk
            self.pos = 0

    def take(self, n):
        self._fill(n)
        b = self.buf[self.pos:self.pos + n]
        self.pos += n
        self.abs += n
        return b

    def u16(self):
        return struct.unpack("<H", self.take(2))[0]

    def u32(self):
        return struct.unpack("<I", self.take(4))[0]

    def u64(self):
        return struct.unpack("<Q", self.take(8))[0]

    def string(self):
        n = self.u64()
        return self.take(n).decode("utf-8", "replace")

    def skip(self, n):
        self._fill(n)
        self.pos += n
        self.abs += n


def parse_gguf(path):
    """Return {tensor_name: [(abs_start, abs_end), ...]} from the dense data section."""
    with open(path, "rb") as fh:
        r = _Reader(fh)
        if r.take(4) != MAGIC:
            raise ValueError("not a GGUF file")
        version = r.u32()
        if version not in (2, 3):
            raise ValueError(f"unsupported gguf version {version}")
        if version == 3:                            # v3: tensor_count first
            tensor_count = r.u64()
            kv_count = r.u64()
        else:                                       # v2: kv_count first
            kv_count = r.u64()
            tensor_count = r.u64()

        def _skip(t, depth=0):                      # value payload by type
            if depth > 8:
                raise ValueError("gguf array nesting too deep (invalid file?)")
            if t in (0, 1, 7):                      # u8/i8/bool -> 1 byte
                r.skip(1)
            elif t in (2, 3):                       # u16/i16
                r.skip(2)
            elif t in (4, 5, 6):                    # u32/i32/f32
                r.skip(4)
            elif t in (10, 11, 12):                 # u64/i64/f64
                r.skip(8)
            elif t == 8:                            # string
                r.skip(r.u64())
            elif t == 9:                            # ARRAY (gguf format)
                et = r.u32()                        # 1) element type  (u32)
                cnt = r.u64()                       # 2) count         (u64)
                if et == 8:                         # fast path: array of strings
                    for _ in range(cnt):
                        r.skip(r.u64())
                else:
                    for _ in range(cnt):
                        _skip(et, depth + 1)
            else:
                raise ValueError(f"unknown gguf metadata type {t}")

        for _ in range(kv_count):
            r.string()                              # key
            _skip(r.u32())                          # value

        infos = []
        for _ in range(tensor_count):
            name = r.string()
            nd = r.u32()
            r.skip(8 * nd)                          # dims (u64 each)
            r.u32()                                 # ggml type
            off = r.u64()                           # offset from data section
            infos.append((name, off))
        data_offset = r.abs

    starts = sorted((data_offset + off, name) for name, off in infos)
    size = os.path.getsize(path)
    ranges = {}
    for i, (start, name) in enumerate(starts):
        end = starts[i + 1][0] if i + 1 < len(starts) else size
        ranges.setdefault(name, []).append((start, end))
    return ranges


# ---------------------------------------------------------------------------
# page-cache inspection (mincore / fadvise via ctypes)
# ---------------------------------------------------------------------------
PAGE = os.sysconf("SC_PAGE_SIZE")
POSIX_FADV_DONTNEED = 4          # Linux

_libc = None
def get_libc():
    global _libc
    if _libc is None:
        _libc = ctypes.CDLL(None, use_errno=True)
        _libc.posix_fadvise.argtypes = [ctypes.c_int, ctypes.c_long,
                                        ctypes.c_long, ctypes.c_int]
        _libc.posix_fadvise.restype = ctypes.c_int
        _libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t,
                                  ctypes.c_void_p]
        _libc.mincore.restype = ctypes.c_int
    return _libc


def sweep_file(path):
    """Drop the whole file from the page cache (cold baseline)."""
    fd = os.open(path, os.O_RDONLY)
    try:
        if get_libc().posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED) != 0:
            raise OSError(ctypes.get_errno(), "posix_fadvise failed")
    finally:
        os.close(fd)
    print(f"sweep: {path} dropped from page cache")


def resident_pages(path):
    """Return (file_size, bytearray resid) -- one byte per page of the file.

    mincore over a MAP_SHARED view reports RESIDENCY IN THE PAGE CACHE of the
    file (not of this process), so it works after the streamer has exited.
    """
    fd = os.open(path, os.O_RDONLY)
    try:
        size = os.fstat(fd).st_size
        pages = size // PAGE
        mm = mmap.mmap(fd, 0, access=mmap.ACCESS_COPY)   # writable buffer (COW), file untouched
        try:
            vec = (ctypes.c_char * pages)()
            addr = ctypes.addressof((ctypes.c_char * 1).from_buffer(mm))
            for _ in range(3):                       # mincore can EAGAIN transiently
                if get_libc().mincore(addr, pages * PAGE, vec) == 0:
                    break
            else:
                raise OSError(ctypes.get_errno(), "mincore failed")
            resid = bytearray(vec)
        finally:
            mm.close()
        return size, resid
    finally:
        os.close(fd)


def group_stats(group_ranges, path):
    """Tensors in group_ranges -> (n_tensors, total_bytes, resident_bytes)."""
    size, resid = resident_pages(path)
    n_pages = len(resid)
    tbytes = rbytes = 0
    for ranges in group_ranges.values():
        tbytes += sum(end - start for start, end in ranges)
        for start, end in ranges:
            p0 = max(0, start // PAGE)
            p1 = min(n_pages, (end + PAGE - 1) // PAGE)
            for p in range(p0, p1):
                if resid[p]:
                    # count only the bytes of THIS range in the resident page
                    ps = max(start, p * PAGE)
                    pe = min(end, (p + 1) * PAGE)
                    rbytes += pe - ps
    return len(group_ranges), tbytes, rbytes


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", help="path to the .gguf")
    ap.add_argument("mode", choices=["sweep", "check"])
    ap.add_argument("--expert-match", default="exps",
                    help="name substring of expert tensors (default: 'exps')")
    ap.add_argument("--control-match", default="attn",
                    help="name substring of control tensors (default: 'attn')")
    args = ap.parse_args()

    if args.mode == "sweep":
        sweep_file(args.model)
        return 0

    print(f"[1/3] parsing header of {os.path.basename(args.model)} ...", file=sys.stderr)
    ranges = parse_gguf(args.model)
    expert = {n: r for n, r in ranges.items() if args.expert_match in n}
    control = {n: r for n, r in ranges.items()
               if args.control_match in n and args.expert_match not in n}
    if not expert:
        print(f"error: no tensors match '{args.expert_match}' (check --expert-match)")
        return 2
    if not control:
        print(f"error: no control tensors match '{args.control_match}'")
        return 2
    print(f"[2/3] mincore over {len(expert)}+{len(control)} tensors ...", file=sys.stderr)

    ne, te, re = group_stats(expert, args.model)
    nc, tc, rc = group_stats(control, args.model)
    size, resid = resident_pages(args.model)
    whole_r = sum(resid) * PAGE

    def pct(a, b):
        return 100.0 * a / b if b else 0.0

    print("== check_cache_purity ==")
    print(f"model:  {args.model}")
    print(f"size:   {size/1e9:.2f} GB ({len(resid)} pages)")
    print(f"{'group':<10}{'tensors':>8}{'total MB':>12}{'resident MB':>14}{'resident%':>11}")
    print("-" * 55)
    print(f"{'expert':<10}{ne:>8}{te/1e6:>12.1f}{re/1e6:>14.1f}{pct(re, te):>10.1f}%")
    print(f"{'control':<10}{nc:>8}{tc/1e6:>12.1f}{rc/1e6:>14.1f}{pct(rc, tc):>10.1f}%")
    print(f"{'whole':<10}{'-':>8}{size/1e6:>12.1f}{whole_r/1e6:>14.1f}{pct(whole_r, size):>10.1f}%")

    ef, cf = pct(re, te), pct(rc, tc)
    if ef < max(50.0, cf * 0.5):
        print("\nverdict: PURE      (expert << control -> DONTNEED is working)")
        print("         the streamer does not depend on the model fitting in RAM.")
        return 0
    print("\nverdict: NOT PURE  (expert ~= control)")
    print("         possible causes:")
    print("         - you ran with dontneed=0 (or --sweep was skipped)")
    print("         - prefill/warm-up reads stay cached (rerun with more tokens)")
    print("         - the A/B protocol was not followed:")
    print("             sweep -> stream(dontneed=1) -> check  vs  sweep -> stream(dontneed=0) -> check")
    return 1


if __name__ == "__main__":
    sys.exit(main())