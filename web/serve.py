#!/usr/bin/env python3
"""Serve the repo root locally:  python3 web/serve.py   then open http://localhost:8793/
(index.html sits at the repo root so GitHub Pages / Vercel serve it at /; assets stay in web/)
Needed because the page fetches binary data files, which browsers refuse over file://."""
import http.server, os, webbrowser
PORT = 8793
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repo root
httpd = http.server.ThreadingHTTPServer(("127.0.0.1", PORT), http.server.SimpleHTTPRequestHandler)
print(f"serving on http://localhost:{PORT}/  (Ctrl-C to stop)")
webbrowser.open(f"http://localhost:{PORT}/")
httpd.serve_forever()
