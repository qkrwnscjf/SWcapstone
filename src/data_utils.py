#!/usr/bin/env python3
"""
Data utilities: Dataset classes, transforms, and data loading helpers.
"""
import os
import csv
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


# ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms(input_size=224):
    """Training transforms with augmentation."""
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_eval_transforms(input_size=224):
    """Evaluation transforms (no augmentation)."""
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_inverse_normalize():
    """Inverse normalization for visualization."""
    return transforms.Normalize(
        mean=[-m / s for m, s in zip(IMAGENET_MEAN, IMAGENET_STD)],
        std=[1.0 / s for s in IMAGENET_STD],
    )


class AnomalyDataset(Dataset):
    """
    Dataset that reads from split CSV files.
    CSV columns: path, dataset_type, defect_type, label, round, split
    """

    def __init__(self, csv_path, transform=None, return_meta=False):
        """
        Args:
            csv_path: Path to split CSV file
            transform: torchvision transforms
            return_meta: If True, return (image, label, meta_dict)
        """
        self.transform = transform
        self.return_meta = return_meta
        self.samples = []

        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                label = 1 if row['label'] == 'anomaly' else 0
                self.samples.append({
                    'path': row['path'],
                    'label': label,
                    'dataset_type': row['dataset_type'],
                    'defect_type': row['defect_type'],
                    'round': row['round'],
                    'split': row['split'],
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample['path']).convert('RGB')

        if self.transform:
            image = self.transform(image)

        label = sample['label']

        if self.return_meta:
            meta = {
                'path': sample['path'],
                'dataset_type': sample['dataset_type'],
                'defect_type': sample['defect_type'],
            }
            return image, label, meta

        return image, label

    def get_class_counts(self):
        """Return count of normal (0) and anomaly (1) samples."""
        labels = [s['label'] for s in self.samples]
        n_normal = labels.count(0)
        n_anomaly = labels.count(1)
        return n_normal, n_anomaly

    def get_pos_weight(self):
        """Calculate pos_weight for BCEWithLogitsLoss (n_neg / n_pos)."""
        n_normal, n_anomaly = self.get_class_counts()
        if n_anomaly == 0:
            raise ValueError("No anomaly samples in dataset! Cannot train gate.")
        return torch.tensor([n_normal / n_anomaly], dtype=torch.float32)


def create_dataloader(csv_path, transform=None, batch_size=32, shuffle=False,
                      num_workers=0, return_meta=False, pin_memory=False):
    """Create a DataLoader from a split CSV."""
    dataset = AnomalyDataset(csv_path, transform=transform, return_meta=return_meta)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    return loader


def get_split_path(splits_dir, round_num, split_name):
    """Get the path to a split CSV file."""
    filename = f"round{round_num}_{split_name}.csv"
    return os.path.join(splits_dir, filename)


def verify_split(csv_path, expected_label=None):
    """
    Verify a split CSV:
    - All paths exist
    - If expected_label set, check all labels match
    Returns (ok, message)
    """
    issues = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if not os.path.exists(row['path']):
                issues.append(f"Row {i}: path not found: {row['path']}")
            if expected_label and row['label'] != expected_label:
                issues.append(f"Row {i}: expected label={expected_label}, got {row['label']}")

    if issues:
        return False, "\n".join(issues[:10])  # first 10 issues
    return True, "OK"


def compute_dataset_stats(csv_path):
    """Compute summary statistics for a split."""
    stats = {
        'total': 0,
        'normal': 0,
        'anomaly': 0,
        'by_dataset': {},
        'by_defect': {},
    }
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            stats['total'] += 1
            if row['label'] == 'normal':
                stats['normal'] += 1
            else:
                stats['anomaly'] += 1

            ds = row['dataset_type']
            if ds not in stats['by_dataset']:
                stats['by_dataset'][ds] = {'normal': 0, 'anomaly': 0}
            stats['by_dataset'][ds][row['label']] += 1

            dt = f"{ds}/{row['defect_type']}"
            if dt not in stats['by_defect']:
                stats['by_defect'][dt] = {'normal': 0, 'anomaly': 0}
            stats['by_defect'][dt][row['label']] += 1

    return stats
