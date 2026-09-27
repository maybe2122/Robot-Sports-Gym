#!/usr/bin/env python3
"""Reference baselines for the tennis return task; see run_fixture_baselines.py.

    python scripts/run_tennis_baselines.py [--out reports] [--split test]
"""

from __future__ import annotations

import sys

from run_fixture_baselines import main

if __name__ == "__main__":
    sys.exit(main(["--sport", "tennis", *sys.argv[1:]]))
