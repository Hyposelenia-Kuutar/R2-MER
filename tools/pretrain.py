import argparse
import os
from pathlib import Path
import sys

import cv2
import torch
import torch.nn as nn
from pytorch_msssim import SSIM
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs import load_config
from tools.generate_mix import prepare_pretrain_data
from models import RegionRecoveryModel


class MicroExprDataset(Dataset):
    def __init__(self, safe_mix_root, orig_data_roots_dict):
        self.valid_pairs = []

        for root, _, files in os.walk(safe_mix_root):
            for file in files:
                if not (
                    file.startswith("mix_")
                    and file.lower().endswith((".jpg", ".png", ".bmp"))
                ):
                    continue

                mix_path = os.path.join(root, file)

                rel_path_full = os.path.relpath(root, safe_mix_root)
                parts = rel_path_full.split(os.sep)

                if len(parts) < 2:
                    continue

                dataset_name = parts[0]
                orig_rel_path = os.path.join(*parts[1:])

                if dataset_name not in orig_data_roots_dict:
                    continue

                orig_root = orig_data_roots_dict[dataset_name]
                orig_path = os.path.join(orig_root, orig_rel_path)

                if os.path.exists(orig_path):
                    self.valid_pairs.append((orig_path, mix_path))

        print(f"Successfully loaded {len(self.valid_pairs)} image pairs.")

    def __len__(self):
        return len(self.valid_pairs)

    def __getitem__(self, idx):
        orig_path, mix_path = self.valid_pairs[idx]
        return {
            "mixed": self._load(mix_path),
            "target": self._load(orig_path),
        }

    def _load(self, path):
        img = cv2.imread(path)

        if img is None:
            raise ValueError(f"OpenCV failed to read: {path}")

        img = cv2.resize(img, (224, 224))

        img = torch.tensor(img).float().permute(2, 0, 1)

        # [0, 255] -> [-1, 1]
        img = img / 127.5 - 1.0

        return img


def parse_args():
    parser = argparse.ArgumentParser(
        "R2-MER Region-Reconstruction Pre-training"
    )

    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "pretrain.py"),
        help="Path to the pre-training config file.",
    )

    # Optional overrides.
    parser.add_argument(
        "--datasets",
        nargs="+",
        type=str,
        default=None,
        help="Optional subset of datasets defined in the config.",
    )
    parser.add_argument("--test_subject", type=str, default=None)
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lambda_l1", type=float, default=None)
    parser.add_argument("--lambda_ssim", type=float, default=None)

    return parser.parse_args()


def main():
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    config = load_config(config_path)

    gpu_ids = str(config.get("gpu_ids", "0"))
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids

    selected_datasets = (
        args.datasets
        if args.datasets is not None
        else config["datasets"]
    )

    data_cfg = config["data"]

    unknown_datasets = [
        name
        for name in selected_datasets
        if name not in data_cfg
    ]
    if unknown_datasets:
        raise KeyError(
            "Datasets are not defined in the pre-training config: "
            + ", ".join(unknown_datasets)
        )

    dataset_configs = []
    orig_data_roots_dict = {}

    for dataset_name in selected_datasets:
        dataset_info = data_cfg[dataset_name]

        dataset_configs.append(
            {
                "name": dataset_name,
                "data_root": dataset_info["rgb_root"],
                "csv_path": dataset_info["csv_path"],
            }
        )

        orig_data_roots_dict[dataset_name] = dataset_info["rgb_root"]

    if args.test_subject is not None:
        actual_subject = (
            None
            if args.test_subject.lower() == "none"
            else args.test_subject
        )
    else:
        actual_subject = config.get("test_subject")

    save_path = (
        args.save_path
        if args.save_path is not None
        else config["save_path"]
    )
    epochs = (
        args.epochs
        if args.epochs is not None
        else int(config["epochs"])
    )
    batch_size = (
        args.batch_size
        if args.batch_size is not None
        else int(config["batch_size"])
    )

    loss_cfg = config["loss"]

    lambda_l1 = (
        args.lambda_l1
        if args.lambda_l1 is not None
        else float(loss_cfg["lambda_l1"])
    )
    lambda_ssim = (
        args.lambda_ssim
        if args.lambda_ssim is not None
        else float(loss_cfg["lambda_ssim"])
    )

    optimizer_cfg = config["optimizer"]
    learning_rate = float(optimizer_cfg["lr"])
    weight_decay = float(optimizer_cfg["weight_decay"])
    grad_clip = float(config.get("grad_clip", 1.0))

    safe_mix_folder = prepare_pretrain_data(
        dataset_configs=dataset_configs,
        save_base=config["mix_save_base"],
        num_classes=int(config["num_classes"]),
        subject=actual_subject,
    )

    dataset = MicroExprDataset(
        safe_mix_folder,
        orig_data_roots_dict,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    if len(dataloader.dataset) == 0:
        print("DataLoader is empty.")
        return

    model = RegionRecoveryModel().cuda()

    if torch.cuda.device_count() > 1:
        print(f"Detected {torch.cuda.device_count()} GPUs.")
        model = nn.DataParallel(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
    )

    criterion_l1 = nn.L1Loss()
    criterion_ssim = SSIM(
        data_range=2.0,
        size_average=True,
        channel=3,
    )

    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    print(f"Config: {config_path}")
    print(f"Datasets: {selected_datasets}")
    print(f"Mixed-image directory: {safe_mix_folder}")
    print(f"Checkpoint: {save_path}")
    print(
        "Loss: "
        f"{lambda_l1} * L1 + "
        f"{lambda_ssim} * (1 - SSIM)"
    )

    for epoch in range(epochs):
        model.train()
        loss_sum = 0.0

        for batch in dataloader:
            optimizer.zero_grad()

            target_img = batch["target"].cuda()
            mixed_img = batch["mixed"].cuda()

            output = model(
                target_img,
                mixed_img,
                True,
            )

            loss_l1 = criterion_l1(
                output,
                target_img,
            )
            loss_ssim = 1 - criterion_ssim(
                output,
                target_img,
            )

            loss = (
                lambda_l1 * loss_l1
                + lambda_ssim * loss_ssim
            )

            if torch.cuda.device_count() > 1:
                loss = loss.mean()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=grad_clip,
            )

            optimizer.step()
            loss_sum += loss.item()

        scheduler.step()

        current_loss = loss_sum / len(dataloader)
        current_lr = scheduler.get_last_lr()[0]

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"LR: {current_lr:.6f} | "
            f"Total Loss: {current_loss:.4f}"
        )

        final_state_dict = (
            model.module.state_dict()
            if isinstance(model, nn.DataParallel)
            else model.state_dict()
        )

        # ---------------------------------------------------------
        # Build checkpoint
        # ---------------------------------------------------------
        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": final_state_dict,
            "loss": current_loss,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        }

        # ---------------------------------------------------------
        # 1. Save latest checkpoint
        #    This file is overwritten after every epoch.
        # ---------------------------------------------------------
        torch.save(
            checkpoint,
            save_path,
        )

        # ---------------------------------------------------------
        # 2. Save an additional checkpoint for the current epoch
        # ---------------------------------------------------------
        checkpoint_dir = os.path.dirname(save_path)

        epoch_checkpoint_path = os.path.join(
            checkpoint_dir,
            f"epoch_{epoch + 1:03d}_loss_{current_loss:.4f}.pth"
        )

        torch.save(
            checkpoint,
            epoch_checkpoint_path,
        )

        print(
            f"Latest checkpoint updated: {save_path}\n"
            f"Epoch checkpoint saved: {epoch_checkpoint_path}"
        )

    print(
        "Pre-training completed. "
        f"Final checkpoint: {save_path}"
    )


if __name__ == "__main__":
    main()