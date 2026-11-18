#!/usr/bin/env python3
"""Local Interactive Web Server for Matchmaking Optimization Demo."""
from __future__ import annotations

import http.server
import os
from pathlib import Path
import socketserver
import sys
import webbrowser

PORT = 8000
REPO_ROOT = Path(__file__).resolve().parent.parent


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(REPO_ROOT), **kwargs)

    def log_message(self, format, *args):
        # Concise logging
        sys.stderr.write(f"[Demo Server] {args[0]} -> {args[1]}\n")


def find_free_port(start_port: int = 8000) -> int:
    import socket
    port = start_port
    while port < start_port + 100:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(('127.0.0.1', port)) != 0:
                return port
        port += 1
    return start_port


def main() -> None:
    port = find_free_port(PORT)
    url = f"http://localhost:{port}/demo/index.html"

    print("=" * 65)
    print(" ⚡ Matchmaking Optimization Engine — Live Interactive Demo")
    print("=" * 65)
    print(f" Serving demo at: {url}")
    print(" Opening browser automatically...")
    print(" Press Ctrl+C to stop the server.")
    print("=" * 65)

    webbrowser.open(url)

    with socketserver.TCPServer(("127.0.0.1", port), Handler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server. Goodbye!")


if __name__ == "__main__":
    main()
