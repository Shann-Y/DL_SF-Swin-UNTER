#!/usr/bin/env python3
"""Verify that a dataset loads correctly and print basic statistics."""

import os
import sys
import argparse
import nibabel as nib
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from data import COVID19Dataset


def check_nifti_file(filepath):
    """Print basic info about a single NIfTI file."""
    try:
        img = nib.load(filepath)
        data = img.get_fdata()

        print(f"\n{'='*60}")
        print(f"File: {os.path.basename(filepath)}")
        print(f"{'='*60}")
        print(f"Shape: {data.shape}")
        print(f"Spacing: {img.header.get_zooms()}")
        print(f"Data type: {data.dtype}")
        print(f"Value range: [{data.min():.2f}, {data.max():.2f}]")
        print(f"Mean: {data.mean():.2f}, Std: {data.std():.2f}")

        unique_values = np.unique(data)
        if len(unique_values) < 10:  # likely a label map
            print(f"Unique values: {unique_values}")
            print(f"  -> This appears to be a LABEL file")
        else:
            print(f"Unique values: {len(unique_values)} (showing first 10: {unique_values[:10]})")
            print(f"  -> This appears to be an IMAGE file")

        return True
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return False


def verify_dataset(data_root, dataset_type="zenodo"):
    """Load the dataset and report per-sample + aggregate statistics."""
    print("\n" + "="*70)
    print(f"VERIFYING {dataset_type.upper()} DATASET")
    print("="*70)
    print(f"Data root: {data_root}\n")

    try:
        dataset = COVID19Dataset(
            data_root=data_root,
            dataset_type=dataset_type,
            split="all",
        )

        print(f"\n[OK] Dataset loaded successfully!")
        print(f"  Total samples: {len(dataset.data_dicts)}")

        print(f"\nFirst 3 samples:")
        for i, item in enumerate(dataset.data_dicts[:3]):
            print(f"\n  Sample {i+1}:")
            print(f"    Image: {item['image']}")
            print(f"    Label: {item['label']}")
            print(f"    [{'OK' if os.path.exists(item['image']) else 'FAIL'}] Image file")
            print(f"    [{'OK' if os.path.exists(item['label']) else 'FAIL'}] Label file")

        print(f"\n{'='*70}")
        print("DETAILED CHECK OF FIRST SAMPLE")
        print("="*70)

        first_sample = dataset.data_dicts[0]
        print("\n[1] Checking IMAGE file:")
        check_nifti_file(first_sample['image'])
        print("\n[2] Checking LABEL file:")
        check_nifti_file(first_sample['label'])

        print(f"\n{'='*70}")
        print("DATASET STATISTICS")
        print("="*70)

        all_shapes = []
        all_spacings = []
        for item in dataset.data_dicts:
            try:
                img = nib.load(item['image'])
                all_shapes.append(img.shape)
                all_spacings.append(img.header.get_zooms())
            except Exception:
                pass

        if all_shapes:
            print(f"\nImage shapes (showing first 5):")
            for i, shape in enumerate(all_shapes[:5]):
                print(f"  {i+1}. {shape}")

            print(f"\nSpacings (showing first 5):")
            for i, spacing in enumerate(all_spacings[:5]):
                print(f"  {i+1}. {spacing}")

        return True

    except Exception as e:
        print(f"\n[FAIL] Failed to load dataset!")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Verify COVID-19 CT dataset")
    parser.add_argument("--data_root", type=str, required=True, help="Root directory of the dataset")
    parser.add_argument("--dataset_type", type=str, default="zenodo",
                        choices=["zenodo", "synthetic"], help="Type of dataset")
    parser.add_argument("--check_file", type=str, default=None, help="Path to a specific NIfTI file to check")

    args = parser.parse_args()

    if args.check_file:
        check_nifti_file(args.check_file)
    else:
        success = verify_dataset(args.data_root, args.dataset_type)
        if success:
            print("\n" + "="*70)
            print("[OK] VERIFICATION COMPLETED SUCCESSFULLY!")
            print("="*70)
            print("\nYour dataset is ready for training!")
            print("\nNext step:")
            print(f"  python train.py --data_root {args.data_root}")
        else:
            print("\n" + "="*70)
            print("[FAIL] VERIFICATION FAILED")
            print("="*70)
            print("\nPlease check the error messages above and fix the issues.")
            sys.exit(1)


if __name__ == "__main__":
    main()
