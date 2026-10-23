"""PyInstaller entry point for the Windows app: the same as `python -m jagimage.desktop`."""
import sys

from jagimage.desktop import main

if __name__ == "__main__":
    sys.exit(main())
