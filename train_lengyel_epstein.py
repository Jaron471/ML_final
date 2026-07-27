#!/usr/bin/env python3
"""Convenience entry point for paired Lengyel--Epstein training."""

import sys

from train_brusselator import parse_args, train


if __name__ == "__main__":
    if "--pde" not in sys.argv:
        sys.argv.extend(["--pde", "lengyel_epstein"])
    train(parse_args())
