"""Luminar Videos standalone: python app.py -> native window, no browser needed."""
import threading
import webview  # pywebview: uses the Edge WebView2 runtime that ships with Windows 11
import server

try:
    srv = server.Server(("127.0.0.1", server.PORT), server.H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
except OSError:
    pass  # a server.py / second window already serves the editor; just open another window onto it
webview.create_window("Luminar Videos", f"http://127.0.0.1:{server.PORT}", width=1500, height=950, min_size=(1100, 700))
webview.start()
