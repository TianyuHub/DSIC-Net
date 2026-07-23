#!/usr/bin/env python3
"""Validate a DSIC-Net manifest and all referenced image/graph files."""

from __future__ import annotations

import argparse
import json

from dsic_net.data import validate_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/demo/manifest.csv")
    args = parser.parse_args()
    print(json.dumps(validate_manifest(args.manifest), indent=2))


if __name__ == "__main__":
    main()
