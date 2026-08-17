#!/usr/bin/env python3
"""
generate_dataset.py
-------------------
Entry point for synthetic dataset generation per Applied Materials problem statement.

Generates paired (Reference Image, Search Image) samples for DRAM-style and
FinFET-style die architectures with recorded ground-truth match locations.

Usage:
    python3 generate_dataset.py --out ./data/self_eval --n 30
    python3 generate_dataset.py --out ./data/ood_holdout --n 10 --styles mixed_logic --seed0 9000

Output:
    <out>/reference/<sample_id>.png    (1000x101000 grayscale, 100x magnification)
    <out>/search/<sample_id>.png      (1000x1000 grayscale, 10x magnification)
    <out>/ground_truth.json           (metadata including GT x,y, seed, style, transforms)

Problem Statement Compliance:
- Reference: 1000x1000, 100x mag, native-resolution crop
- Search: 1000x1000, 10x mag, wider FOV containing target
- Scale: nominal 10:1, robustness tested at ~9:1 to 11:1
- Independent per-image degradation (separate RNGs for noise/blur/rotation)
- Metadata recorded: seed, architecture, transforms, noise settings, scale, rotation
- Literature-supported DRAM/FinFET structures with citations
"""

import sys
import os

# Add src to path for module imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from dataset.build_dataset import main

if __name__ == "__main__":
    main()