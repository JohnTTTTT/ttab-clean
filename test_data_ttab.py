# test_data_ttab_simple.py

import argparse
from torch.utils.data import DataLoader
from ttab.loads.datasets.datasets import AffectNetDataset
from ttab.loads.datasets.utils.preprocess_toolkit import get_transform

def get_args():
    p = argparse.ArgumentParser("TTAB Data Loader Smoke Test")
    p.add_argument("--data_root",   required=True,
                   help="Root path to AffectNet7_37k_balanced")
    p.add_argument("--batch_size",  type=int, default=16)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--pin_mem",     action="store_true",
                   help="Pin memory in DataLoader")
    p.add_argument("--input_size",  type=int, default=224,
                   help="Image size for transform")
    return p.parse_args()

def main():
    args = get_args()

    # build transform & dataset
    transform = get_transform(name="affectnet", input_size=args.input_size, augment=False)
    ds = AffectNetDataset(
        root=args.data_root,
        split="val",
        transform=transform
    )
    print(f"Dataset size: {len(ds)} samples")

    # build loader
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    # grab one batch
    imgs, labels = next(iter(loader))
    print(f"Batch images shape : {tuple(imgs.shape)}")
    print(f"Batch labels shape : {tuple(labels.shape)}")
    print(f"Unique labels in batch: {set(labels.tolist())}")

if __name__ == "__main__":
    main()
