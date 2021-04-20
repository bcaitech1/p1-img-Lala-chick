import torch
from torch import nn
import os
import random
import torchvision
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse
import json

from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F

from sklearn.metrics import f1_score

import sklearn
from sklearn import metrics
from data import custom_cutmix, prepare_train_val_loader
from models import *
from loss import CeF1Loss


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_folder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except:
        pass


def train_one_epoch(
    epoch,
    scaler,
    model,
    loss_fn,
    optimizer,
    train_loader,
    old_train_loader,
    device,
    scheduler,
    cfg,
):
    model.train()

    running_loss = None
    old_iter = iter(old_train_loader)
    old_length = len(old_train_loader)
    old_step = 0
    pbar = tqdm(
        enumerate(train_loader), total=len(train_loader), position=0, leave=True
    )

    for step, (imgs, image_labels) in pbar:
        imgs = imgs.to(device).float()
        image_labels = image_labels.to(device).long()
        mix_decision = np.random.rand()
        if mix_decision < cfg["mix_prob"]:
            if old_step < old_length - 1:
                old_imgs, old_labels = next(old_iter)
                old_imgs = old_imgs.to(device).float()
                old_labels = old_labels.to(device).long()
                old_step += 1
            else:
                old_step = 1
                old_iter = iter(old_train_loader)
                old_imgs, old_labels = next(old_iter)
                old_imgs = old_imgs.to(device).float()
                old_labels = old_labels.to(device).long()
            imgs, image_labels = custom_cutmix(imgs, image_labels, old_imgs, old_labels)

        with autocast():
            image_preds = model(imgs.float())

            if mix_decision < cfg["mix_prob"]:
                loss = loss_fn(image_preds, image_labels[0]) * image_labels[
                    2
                ] + loss_fn(image_preds, image_labels[1]) * (1.0 - image_labels[2])
                loss /= cfg["gradient_accumulation_steps"]
            else:
                loss = loss_fn(image_preds, image_labels)
                loss /= cfg["gradient_accumulation_steps"]

            scaler.scale(loss).backward()

            if running_loss is None:
                running_loss = loss.item()
            else:
                running_loss = running_loss * 0.99 + loss.item() * 0.01

            if ((step + 1) % cfg["gradient_accumulation_steps"] == 0) or (
                (step + 1) == len(train_loader)
            ):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                description = f"epoch {epoch} loss: {running_loss:.4f}"
                pbar.set_description(description)
    scheduler.step()


def valid_one_epoch(epoch, model, loss_fn, val_loader, device):
    model.eval()

    loss_sum = 0
    sample_num = 0
    image_preds_all = []
    image_targets_all = []

    pbar = tqdm(enumerate(val_loader), total=len(val_loader), position=0, leave=True)
    for step, (imgs, image_labels) in pbar:
        imgs = imgs.to(device).float()
        image_labels = image_labels.to(device).long()

        image_preds = model(imgs)

        image_preds_all += [torch.argmax(image_preds, 1).detach().cpu().numpy()]
        image_targets_all += [image_labels.detach().cpu().numpy()]

        loss = loss_fn(image_preds, image_labels)
        loss_sum += loss.item() * image_labels.shape[0]
        sample_num += image_labels.shape[0]

        if ((step + 1) % 1 == 0) or ((step + 1) == len(val_loader)):
            description = f"epoch {epoch} loss: {loss_sum/sample_num: .4f}"
            pbar.set_description(description)

    image_preds_all = np.concatenate(image_preds_all)
    image_targets_all = np.concatenate(image_targets_all)
    accuracy = (image_preds_all == image_targets_all).mean()
    f1 = f1_score(image_preds_all, image_targets_all, average="macro")
    print(f"validation accuracy = {accuracy: .4f}, f1 score = {f1: .4f}")

    return accuracy, f1


def main(cfg):
    train = pd.read_csv("./train.csv")
    old = train[train["age"] == 2]
    seed_everything(cfg["seed"])
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    for fold in range(cfg["fold_num"]):
        train_loader, val_loader = prepare_train_val_loader(train, [fold], cfg)
        old_train_loader, old_val_loader = prepare_train_val_loader(old, [fold], cfg)
        if cfg["base"] == "efficient":
            model = MaskEfficientNet(
                cfg["model"], train.label.nunique(), pretrained=True
            ).to(device)
        elif cfg["base"] == "nfnet":
            model = MaskNFNet(cfg["model"], train.label.nunique(), pretrained=True).to(
                device
            )
        elif cfg["base"] == "ViT":
            model = MaskViT(cfg["model"], train.label.nunique(), pretrained=True).to(
                device
            )

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg["learning_rate"],
            weight_decay=cfg["weight_decay"],
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=cfg["T_0"], T_mult=1, eta_min=cfg["min_lr"], last_epoch=-1
        )

        loss_fn = CeF1Loss().to(device)

        scaler = GradScaler()

        best_f1 = 0
        best_epoch = 0
        stop_count = 0

        for epoch in range(cfg["epochs"]):
            train_one_epoch(
                epoch,
                scaler,
                model,
                loss_fn,
                optimizer,
                train_loader,
                old_train_loader,
                device,
                scheduler,
                cfg,
            )

            with torch.no_grad():
                epoch_acc, epoch_f1 = valid_one_epoch(
                    epoch, model, loss_fn, val_loader, device
                )

            if epoch_f1 > best_f1:
                stop_count = 0
                dir = f"./trained_models/{cfg['base']}/{cfg['model']}"
                create_folder(dir)
                torch.save(model.state_dict(), f"{dir}/{cfg['model']}_{fold}.pth")

                best_f1 = epoch_f1
                best_epoch = epoch
                print("The model is saved!")

            else:
                stop_count += 1
                if stop_count > cfg["early_stop"]:
                    break

        del model, optimizer, train_loader, val_loader, scaler, scheduler
        torch.cuda.empty_cache()
        print(f"Best F1: {best_f1} in epoch {best_epoch}\n")


if __name__ == "__main__":
    cli_parser = argparse.ArgumentParser()
    cli_parser.add_argument("--model", type=str, default="tf_efficientnet_b3_ns")
    cli_parser = cli_parser.parse_args()
    cfg = None
    if cli_parser.model == "tf_efficientnet_b3_ns":
        f = open("./config/effnet_config.json", encoding="UTF-8")
        cfg = json.loads(f.read())
    elif cli_parser.model == "eca_nfnet_l0":
        f = open("./config/nfnet_config.json", encoding="UTF-8")
        cfg = json.loads(f.read())
    elif cli_parser.model == "vit_base_patch16_384":
        f = open("./config/vit_config.json", encoding="UTF-8")
        cfg = json.loads(f.read())
    main(cfg)
