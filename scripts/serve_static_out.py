import http.server
import socketserver
from pathlib import Path
import urllib.parse

PORT = 3000
OUT_DIR = Path(__file__).resolve().parent.parent / "frontend" / "out"


class CleanUrlHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(OUT_DIR), **kwargs)

    def translate_path(self, path):
        # Parse path and query
        parsed = urllib.parse.urlparse(path)
        clean = parsed.path.strip("/")

        if clean == "":
            target = OUT_DIR / "index.html"
            if target.exists():
                return str(target)
            return str(OUT_DIR)

        # 1. Exact file match
        direct = OUT_DIR / clean
        if direct.exists() and not direct.is_dir():
            return str(direct)

        # 2. Directory with index.html
        as_dir = OUT_DIR / clean / "index.html"
        if as_dir.exists():
            return str(as_dir)

        # 3. Clean .html file
        as_html = OUT_DIR / f"{clean}.html"
        if as_html.exists():
            return str(as_html)

        return super().translate_path(path)


if __name__ == "__main__":
    socketserver.ThreadingTCPServer.allow_reuse_address = True
    with socketserver.ThreadingTCPServer(("", PORT), CleanUrlHandler) as httpd:
        print(f"Serving {OUT_DIR} at http://localhost:{PORT}")
        httpd.serve_forever()
