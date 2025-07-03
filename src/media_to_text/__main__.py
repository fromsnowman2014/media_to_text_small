"""
Main entry point for the media-to-text converter.
Allows running the package directly with `python -m media_to_text`
"""

import sys
from .cli.cli import main

if __name__ == "__main__":
    sys.exit(main())
