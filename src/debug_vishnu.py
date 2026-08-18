#!/usr/bin/env python3
"""
Debug V6 on Vishnu's dataset - understand failure modes.
"""

import sys
import os
import json
import numpy as np
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.localization.enhanced_localizer import localize_v6, ncc_map, get_feature_map, extract_nms_peaks, subpixel_refine_2d


def main():
    sample_id = 0
    ref_path = f"/tmp/competitor_analysis/vishnu_drift-sense/test_dataset/reference/{sample_id:04d}.png"
    search_path = f"/tmp/competitor_analysis/vishnu_drift-sense/test_dataset/search/{sample_id:04d}.png"

    ref_img = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
    search_img = cv2.imread(search_path, cv2.IMREAD_GRAYSCALE)

    # Run with debug
    result = localize_v6(ref_img, search_img, nominal_downsample=10.0, return_debug=True)

    print(f"Result: {result}")
    print(f"\nDebug info:")
    if 'debug' in result:
        for k, v in result['debug'].items():
            if k == 'top_candidates':
                print(f"  {k}:")
                for i, c in enumerate(v):
                    print(f"    {i}: x={c['x']:.1f}, y={c['y']:.1f}, score={c['score']:.4f}, phase={c.get('phase_response', 'N/A')}")
            else:
                print(f"  {k}: {v}")


if __name__ == "__main__":
    main()