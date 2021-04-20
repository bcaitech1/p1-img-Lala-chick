import cv2
import pandas as pd
from torch.utils.data import DataLoader
from .dataset import MaskDataset, MaskTestDataset
from .augmentations import (
    get_train_transforms_distortion,
    get_train_transforms_no_distortion,
    get_test_transforms,
)


def prepare_train_val_loader(df, fold, cfg):

    train = df[~df.groups.isin(fold)].reset_index(drop=True)
    valid = df[df.groups.isin(fold)].reset_index(drop=True)

    if cfg["distortion"]:
        train_ds = MaskDataset(
            train, transforms=get_train_transforms_distortion(cfg), output_label=True
        )
    else:
        train_ds = MaskDataset(
            train, transforms=get_train_transforms_no_distortion(cfg), output_label=True
        )

    valid_ds = MaskDataset(
        valid, transforms=get_test_transforms(cfg), output_label=True
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        pin_memory=True,
        drop_last=False,
        shuffle=True,
        num_workers=cfg["num_workers"],
    )

    valid_loader = DataLoader(
        valid_ds,
        batch_size=cfg["batch_size"],
        pin_memory=True,
        shuffle=False,
        num_workers=cfg["num_workers"],
    )

    return train_loader, valid_loader


def prepare_test_loader(
    df, img_size=384, batch_size=64, data_root="./input/data/eval/images"
):
    df = df.copy()
    test_ds = MaskTestDataset(
        df, transforms=get_test_transforms(img_size), data_root=data_root
    )

    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, pin_memory=False
    )
    return test_loader
