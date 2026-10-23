"""Jag Studio desktop app (the Windows Start menu / desktop shortcut runs this).

Starts the Studio on this computer only (127.0.0.1), opens it in the default browser and keeps
running until this window is closed. Photos go to Documents\\Jag Studio; API keys the user chooses
to remember are kept in their own profile (%APPDATA%\\Jag Studio\\keys.json).
"""

import argparse
import json
import os
import socket
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path

APP = "Jag Studio"
PORTS = range(8765, 8785)


def documents() -> Path:
    """The user's Documents folder (Windows may move it, e.g. into OneDrive)."""
    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes

            buf = ctypes.create_unicode_buffer(wintypes.MAX_PATH)
            if ctypes.windll.shell32.SHGetFolderPathW(None, 5, None, 0, buf) == 0 and buf.value:  # CSIDL_PERSONAL
                return Path(buf.value)
        except (AttributeError, OSError):
            pass
    return Path.home() / "Documents"


def settings_dir() -> Path:
    base = os.environ.get("APPDATA") if os.name == "nt" else os.environ.get("XDG_CONFIG_HOME")
    return Path(base or Path.home() / ".config") / APP


def running_studio(port: int) -> bool:
    """A Jag Studio already answers on this port (e.g. the shortcut was clicked twice)."""
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/options", timeout=2) as r:
            return "categories" in json.loads(r.read())
    except (OSError, ValueError):
        return False


def free_port() -> int:
    for port in PORTS:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise SystemExit(f"Ports {PORTS.start}-{PORTS.stop - 1} are all in use; close other programs and try again.")


def engines_work() -> bool:
    """The packaged app has what a batch needs: the AI clients load and a recolour runs."""
    import numpy as np
    from PIL import Image

    import google.genai  # noqa: F401
    import openai  # noqa: F401

    from .recolor import recolor_image

    img = np.full((60, 40, 3), 237, np.uint8)
    img[10:50, 8:32] = (143, 211, 168)  # a mint "garment"
    out, de, _ = recolor_image(Image.fromarray(img), "#8FD3A8", "#E31B23")
    return de < 3 and np.asarray(out.convert("RGB"))[30, 20, 0] > 180


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog=APP)
    ap.add_argument("--out", help="working folder for photos (default: Documents\\Jag Studio)")
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="start, check the app answers, then exit (installer test)")
    a = ap.parse_args(argv)

    for port in PORTS:
        if running_studio(port) and not a.smoke:
            print(f"{APP} is already running: opening http://127.0.0.1:{port}")
            webbrowser.open(f"http://127.0.0.1:{port}")
            return 0

    from .server import load_keys, serve

    out = Path(a.out) if a.out else documents() / APP
    out.mkdir(parents=True, exist_ok=True)
    keys_file = settings_dir() / "keys.json"
    load_keys(keys_file)
    port = free_port()
    _, httpd = serve("127.0.0.1", port, str(out), keys_file=keys_file)
    url = f"http://127.0.0.1:{port}"
    threading.Thread(target=httpd.serve_forever, daemon=True).start()

    if a.smoke:
        with urllib.request.urlopen(url + "/", timeout=10) as r:
            page = r.read().decode("utf-8", "replace")
        ok = "Jag Studio" in page and running_studio(port) and engines_work()
        print(f"smoke test {'passed' if ok else 'FAILED'}: {url}, photos in {out}")
        httpd.shutdown()
        return 0 if ok else 1

    print(f"{APP} is running at {url}\nPhotos are saved in {out}\n\nKeep this window open while you work; close it to stop {APP}.")
    if not a.no_browser:
        webbrowser.open(url)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass
    httpd.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
