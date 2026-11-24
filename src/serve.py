"""
Lightweight zero-dependency HTTP server with Gzip compression + Cache-Control headers.
Cuts first-load transfer size of JSON/WASM/INT8 weights (~35-75% smaller) and enables
instant browser disk caching on reload.
"""
import gzip
import os
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

ROOT = "/home/antigravity/napkin-ml"
COMPRESS_EXTS = {".html", ".js", ".json", ".wasm", ".bin", ".svg", ".css"}
_GZIP_CACHE = {}


class FastStaticHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=ROOT, **kwargs)

    def end_headers(self):
        # Cache immutable model exports aggressively; revalidate HTML/JS cleanly
        if "/exports/" in self.path:
            self.send_header("Cache-Control", "public, max-age=86400")
        else:
            self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def do_GET(self):
        ae = self.headers.get("Accept-Encoding", "")
        path = self.translate_path(self.path)
        ext = os.path.splitext(path)[1].lower()
        if "gzip" in ae and os.path.isfile(path) and ext in COMPRESS_EXTS:
            mtime = os.path.getmtime(path)
            cached = _GZIP_CACHE.get(path)
            if not cached or cached[0] != mtime:
                with open(path, "rb") as f:
                    raw = f.read()
                comp = gzip.compress(raw, compresslevel=6)
                _GZIP_CACHE[path] = (mtime, comp)
            else:
                comp = cached[1]
            ctype = self.guess_type(path)
            if ext == ".wasm":
                ctype = "application/wasm"
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Encoding", "gzip")
            self.send_header("Content-Length", str(len(comp)))
            self.end_headers()
            self.wfile.write(comp)
            return
        super().do_GET()

    def log_message(self, fmt, *args):
        pass


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    srv = ThreadingHTTPServer(("0.0.0.0", port), FastStaticHandler)
    print(f"Serving {ROOT} with gzip + Cache-Control on http://0.0.0.0:{port}", flush=True)
    srv.serve_forever()
