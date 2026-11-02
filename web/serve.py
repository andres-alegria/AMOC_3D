#!/usr/bin/env python3
"""Serve this folder locally:  python3 serve.py   then open http://localhost:8793/
Needed because the page fetches binary data files, which browsers refuse over file://."""
import http.server, os, webbrowser
PORT = 8793
os.chdir(os.path.dirname(os.path.abspath(__file__)))
httpd = http.server.ThreadingHTTPServer(("127.0.0.1", PORT), http.server.SimpleHTTPRequestHandler)
print(f"serving on http://localhost:{PORT}/  (Ctrl-C to stop)")
webbrowser.open(f"http://localhost:{PORT}/")
httpd.serve_forever()
