#!/usr/bin/env python3
"""Generate manifest.csv consolidating ground_truth.json + predictions.

Per problem statement: reference path, search path, ground-truth x/y, predicted x/y,
per-pair metadata (seed, architecture, transforms, noise, scale, rotation).
"""

import json
import csv
import os
import sys

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def generate_manifest(gt_path, pred_path, output_path):
    gt = load_json(gt_path)
    pred = load_json(pred_path)

    # Expected metadata from ground_truth.json
    # style, seed, ref_native_size, search_size, gt_center_xy, gt_inset_size_px

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'sample_id',
            'style',
            'seed',
            'reference_path',
            'search_path',
            'gt_x',
            'gt_y',
            'pred_x',
            'pred_y',
            'error_px',
            'confidence',
            'ambiguity_ratio',
            'scale_pred',
            'rotation_deg_pred',
            'low_confidence_flag',
            'method',
            # Metadata
            'ref_native_size',
            'search_size_w',
            'search_size_h',
            'gt_inset_size_px',
        ])

        for sample_id, meta in gt.items():
            if sample_id not in pred:
                continue
            p = pred[sample_id]

            gt_x, gt_y = meta['gt_center_xy']
            pred_x = p.get('pred_x', p.get('x', 0))
            pred_y = p.get('pred_y', p.get('y', 0))

            writer.writerow([
                sample_id,
                meta.get('style', ''),
                meta.get('seed', ''),
                f"reference/{sample_id}.png",
                f"search/{sample_id}.png",
                round(gt_x, 2),
                round(gt_y, 2),
                round(pred_x, 2),
                round(pred_y, 2),
                round(p.get('error_px', ((pred_x - gt_x)**2 + (pred_y - gt_y)**2)**0.5), 2),
                p.get('confidence', ''),
                p.get('ambiguity_ratio', ''),
                p.get('scale', ''),
                p.get('rotation_deg', ''),
                p.get('low_confidence_flag', ''),
                p.get('method', ''),
                meta.get('ref_native_size', ''),
                meta.get('search_size', [0,0])[0] if isinstance(meta.get('search_size'), list) else meta.get('search_size', ''),
                meta.get('search_size', [0,0])[1] if isinstance(meta.get('search_size'), list) else '',
                meta.get('gt_inset_size_px', ''),
            ])

    print(f"Manifest written to {output_path}")

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--gt', required=True, help='ground_truth.json path')
    ap.add_argument('--pred', required=True, help='predictions.json path')
    ap.add_argument('--out', required=True, help='Output manifest.csv path')
    args = ap.parse_args()

    generate_manifest(args.gt, args.pred, args.out)

if __name__ == '__main__':
    main()