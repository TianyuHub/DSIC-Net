#!/usr/bin/env python3
"""Generate deterministic synthetic inputs for the software smoke test."""

from __future__ import annotations

import argparse

from dsic_net.demo_data import generate_demo_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="data/demo")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    manifest = generate_demo_dataset(args.output_dir, overwrite=args.overwrite)
    print(manifest)


if __name__ == "__main__":
    main()
