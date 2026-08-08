import argparse
import os
import random
import time
import warnings
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import ConcatDataset, DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs import load_config
from datasets import Fusion_CASME2_Dataset, Fusion_SAMM_Dataset, Fusion_SMIC_Dataset
from models import FusionHTNet


warnings.filterwarnings("ignore", category=UserWarning, module="torch.utils.checkpoint")
warnings.filterwarnings("ignore", message="None of the inputs have requires_grad=True")

def parse_option():
    parser = argparse.ArgumentParser("R2-MER Training")

    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "casme2_3class.py"),
        help="Path to a dataset config file.",
    )
    parser.add_argument("--loso_subject", type=str, required=True)
    parser.add_argument(
        "--preds_save_path",
        type=str,
        required=True,
        help="Path used to save predictions from the best UF1 epoch.",
    )
    parser.add_argument(
        "--weights_save_dir",
        type=str,
        default=None,
        help="Optional supervised checkpoint directory.",
    )

    # Optional command-line overrides.
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--num_classes", type=int, default=None)
    parser.add_argument("--train_ratio", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--recovery_weights", type=str, default=None)

    return parser.parse_args()


def get_default_from_grid(config, key):
    values = config.get("param_grid", {}).get(key)
    if not values:
        raise KeyError(f"Missing '{key}' in config['param_grid'].")
    return values[0]


def resolve_runtime_args(args, config):
    args.dataset = config["dataset"]

    args.batch_size = (
        args.batch_size
        if args.batch_size is not None
        else get_default_from_grid(config, "batch_size")
    )
    args.epochs = (
        args.epochs
        if args.epochs is not None
        else get_default_from_grid(config, "epochs")
    )
    args.lr = (
        args.lr
        if args.lr is not None
        else get_default_from_grid(config, "learning_rate")
    )
    args.num_classes = (
        args.num_classes
        if args.num_classes is not None
        else get_default_from_grid(config, "num_classes")
    )
    args.seed = (
        args.seed
        if args.seed is not None
        else get_default_from_grid(config, "seed")
    )
    args.weight_decay = (
        args.weight_decay
        if args.weight_decay is not None
        else get_default_from_grid(config, "weight_decay")
    )
    args.train_ratio = (
        args.train_ratio
        if args.train_ratio is not None
        else config.get("train_ratio", 1.0)
    )
    args.recovery_weights = (
        args.recovery_weights
        if args.recovery_weights is not None
        else config.get("pretrained_weights", "")
    )

    if args.weights_save_dir is None:
        args.weights_save_dir = "./weights"

    return args


def collate_fn_safe(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        return torch.tensor([]), torch.tensor([]), torch.tensor([])
    return torch.utils.data.dataloader.default_collate(batch)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_single_dataset(config, dataset_name, loso_subject, num_classes):
    data_cfg = config["data"]

    dataset_map = {
        "casme2": Fusion_CASME2_Dataset,
        "samm": Fusion_SAMM_Dataset,
        "smic": Fusion_SMIC_Dataset,
    }

    if dataset_name not in dataset_map:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    dataset_class = dataset_map[dataset_name]

    common_kwargs = {
        "csv_path": data_cfg["csv_path"],
        "flow_features_root": data_cfg["flow_root"],
        "data_root": data_cfg["rgb_root"],
        "original_image_root": data_cfg["rgb_root"],
        "loso_subject": loso_subject,
        "num_classes": num_classes,
    }

    train_ds = dataset_class(mode="train", **common_kwargs)
    val_ds = dataset_class(mode="test", **common_kwargs)

    return train_ds, val_ds


def build_three_db_dataset(config, loso_subject, num_classes):
    try:
        loso_db, sub_id = loso_subject.split("_", 1)
    except ValueError as exc:
        raise ValueError(
            "For dataset='3db', --loso_subject must use '<dataset>_<subject>', "
            "for example 'casme2_01'."
        ) from exc

    if loso_db not in {"casme2", "samm", "smic"}:
        raise ValueError(
            f"Invalid target dataset '{loso_db}'. "
            "Expected one of: casme2, samm, smic."
        )

    datasets_cfg = config["data"]["datasets"]

    c_cfg = datasets_cfg["casme2"]
    s_cfg = datasets_cfg["samm"]
    sm_cfg = datasets_cfg["smic"]

    train_list = [
        Fusion_CASME2_Dataset(
            csv_path=c_cfg["csv_path"],
            flow_features_root=c_cfg["flow_root"],
            data_root=c_cfg["rgb_root"],
            original_image_root=c_cfg["rgb_root"],
            mode="train",
            loso_subject=sub_id if loso_db == "casme2" else None,
            num_classes=num_classes,
        ),
        Fusion_SAMM_Dataset(
            csv_path=s_cfg["csv_path"],
            flow_features_root=s_cfg["flow_root"],
            data_root=s_cfg["rgb_root"],
            original_image_root=s_cfg["rgb_root"],
            mode="train",
            loso_subject=sub_id if loso_db == "samm" else None,
            num_classes=num_classes,
        ),
        Fusion_SMIC_Dataset(
            csv_path=sm_cfg["csv_path"],
            flow_features_root=sm_cfg["flow_root"],
            data_root=sm_cfg["rgb_root"],
            original_image_root=sm_cfg["rgb_root"],
            mode="train",
            loso_subject=sub_id if loso_db == "smic" else None,
            num_classes=num_classes,
        ),
    ]

    train_ds = ConcatDataset(train_list)

    target_cfg = datasets_cfg[loso_db]
    target_class = {
        "casme2": Fusion_CASME2_Dataset,
        "samm": Fusion_SAMM_Dataset,
        "smic": Fusion_SMIC_Dataset,
    }[loso_db]

    val_ds = target_class(
        csv_path=target_cfg["csv_path"],
        flow_features_root=target_cfg["flow_root"],
        data_root=target_cfg["rgb_root"],
        original_image_root=target_cfg["rgb_root"],
        mode="test",
        loso_subject=sub_id,
        num_classes=num_classes,
    )

    return train_ds, val_ds


def subsample_training_set(train_ds, train_ratio, seed):
    if train_ratio >= 1.0:
        return train_ds

    print(f"Training-set subsampling enabled: {train_ratio * 100:.2f}%")

    indices = list(range(len(train_ds)))

    if hasattr(train_ds, "labels"):
        y_train = train_ds.labels
    else:
        print("Extracting labels for stratified sampling...")
        y_train = [train_ds[i][1] for i in indices]

    if not indices:
        return train_ds

    try:
        train_idx, _ = train_test_split(
            indices,
            train_size=train_ratio,
            stratify=y_train,
            random_state=seed,
        )
    except ValueError:
        print(
            "Stratified sampling is unavailable for this split; "
            "falling back to random sampling."
        )
        train_idx, _ = train_test_split(
            indices,
            train_size=train_ratio,
            random_state=seed,
        )

    sampled_ds = Subset(train_ds, train_idx)
    print(f"Training samples after subsampling: {len(sampled_ds)}")
    return sampled_ds


def main():
    args = parse_option()
    config_path = Path(args.config).expanduser().resolve()
    config = load_config(config_path)
    args = resolve_runtime_args(args, config)

    # When train.py is launched directly, use the physical GPU selected
    # in the dataset config. main.py already sets CUDA_VISIBLE_DEVICES
    # before spawning this process, so that setting is preserved.
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config.get("gpu_id", "0"))

    if args.gpu is None:
        args.gpu = 0

    set_seed(args.seed)

    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    )

    os.makedirs(args.weights_save_dir, exist_ok=True)
    weight_path = os.path.join(
        args.weights_save_dir,
        f"{args.loso_subject}.pth",
    )

    print(f"--- Running LOSO for Subject {args.loso_subject} ---")
    print(f"--- Config: {config_path} ---")
    print(f"--- Supervised checkpoint path: {weight_path} ---")

    if args.dataset == "3db":
        train_ds, val_ds = build_three_db_dataset(
            config,
            args.loso_subject,
            args.num_classes,
        )
    else:
        train_ds, val_ds = build_single_dataset(
            config,
            args.dataset,
            args.loso_subject,
            args.num_classes,
        )

    train_ds = subsample_training_set(
        train_ds,
        args.train_ratio,
        args.seed,
    )

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % (2 ** 32)
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    generator = torch.Generator()
    generator.manual_seed(args.seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn_safe,
        worker_init_fn=seed_worker,
        generator=generator,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=config.get("val_batch_size", 256),
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn_safe,
        worker_init_fn=seed_worker,
        generator=generator,
    )

    htnet_config = dict(config["htnet"])
    htnet_config["num_classes"] = args.num_classes

    model = FusionHTNet(
        htnet_config=htnet_config,
        new_module_path=args.recovery_weights,
    )
    model = model.to(device)

    if hasattr(model, "region_recovery"):

        print(
            "Freezing pretrained HKPFE and IRSA; "
            "keeping RGB projection trainable..."
        )

        # Freeze the entire pretrained recovery branch first
        for param in model.region_recovery.parameters():
            param.requires_grad = False

        # The 1x1 projection is NOT used during pre-training,
        # therefore it must be learned during supervised training.
        for param in model.region_recovery.conv.parameters():
            param.requires_grad = True

    unfreeze_epoch = int(config.get("unfreeze_epoch", 100))

    if hasattr(model, "htnet"):
        print(
            "Initial freezing of HTNet parameters "
            f"(will unfreeze at epoch {unfreeze_epoch})..."
        )
        for name, param in model.htnet.named_parameters():
            if name.startswith("mlp_head"):
                param.requires_grad = True
            else:
                param.requires_grad = False

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    best_uf1 = 0.0
    best_acc_at_best_uf1 = 0.0

    subject_start_time = time.time()
    epoch = 1
    total_epochs = args.epochs

    while epoch <= total_epochs:
        epoch_start_time = time.time()

        if epoch == unfreeze_epoch:
            print(f"Epoch {epoch}: Unfreezing HTNet parameters...")
            unfreeze_params = []

            for name, param in model.named_parameters():
                if "htnet" in name and not param.requires_grad:
                    param.requires_grad = True
                    unfreeze_params.append(param)

            if unfreeze_params:
                optimizer.add_param_group(
                    {
                        "params": unfreeze_params,
                        "lr": args.lr,
                    }
                )
                print(f"Added {len(unfreeze_params)} params to optimizer.")

        model.train()
        # Keep the pretrained HKPFE and IRSA in evaluation mode.
        # The supervised RGB projection remains trainable.
        model.region_recovery.eval()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_data in train_loader:
            if len(batch_data) == 0:
                continue

            inputs, label, _ = batch_data
            x1, x2, x3 = inputs

            x1 = x1.to(device)
            x2 = x2.to(device)
            x3 = x3.to(device)
            label = label.to(device)

            optimizer.zero_grad()

            outputs = model(x1, x2, x3)
            loss = criterion(outputs, label)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x1.size(0)
            _, pred = torch.max(outputs, 1)
            train_correct += (pred == label).sum().item()
            train_total += label.size(0)

        train_avg_loss = (
            train_loss / len(train_loader.dataset)
            if len(train_loader.dataset) > 0
            else 0
        )
        train_acc = (
            train_correct / train_total
            if train_total > 0
            else 0
        )

        print(
            f"Sub {args.loso_subject} | "
            f"Epoch {epoch}/{total_epochs} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Loss: {train_avg_loss:.4f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.6f}"
        )

        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch_data in val_loader:
                if len(batch_data) == 0:
                    continue

                inputs, label, _ = batch_data
                x1, x2, x3 = inputs

                x1 = x1.to(device)
                x2 = x2.to(device)
                x3 = x3.to(device)
                label = label.to(device)

                outputs = model(x1, x2, x3)
                loss = criterion(outputs, label)

                val_loss += loss.item() * x1.size(0)

                _, pred = torch.max(outputs, 1)
                val_correct += (pred == label).sum().item()
                val_total += label.size(0)

                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(label.cpu().numpy())

        if val_total > 0:
            val_acc = val_correct / val_total
            val_avg_loss = val_loss / len(val_loader.dataset)
            val_uf1 = f1_score(
                all_targets,
                all_preds,
                average="macro",
            )

            print(
                f"Sub {args.loso_subject} | "
                f"Epoch {epoch}/{total_epochs} | "
                f"Val Acc: {val_acc:.4f} | "
                f"Val UF1: {val_uf1:.4f} | "
                f"Val Loss: {val_avg_loss:.4f}"
            )

            if val_uf1 > best_uf1:
                best_uf1 = val_uf1
                best_acc_at_best_uf1 = val_acc

                save_dir = os.path.dirname(args.preds_save_path)
                if save_dir:
                    os.makedirs(save_dir, exist_ok=True)

                torch.save(
                    {
                        "preds": torch.tensor(all_preds),
                        "targets": torch.tensor(all_targets),
                    },
                    args.preds_save_path,
                )

                # Enable this line if supervised model checkpoints are required.
                # torch.save(model.state_dict(), weight_path)

                if val_uf1 == 1.0:
                    break

        epoch_duration = time.time() - epoch_start_time
        elapsed_time = time.time() - subject_start_time
        avg_time = elapsed_time / epoch if epoch > 0 else 0
        remaining = (total_epochs - epoch) * avg_time

        print(
            f"Time: {epoch_duration:.2f}s | "
            f"Remaining: {remaining:.2f}s\n"
        )

        if train_acc >= 0.9999 and 150 < epoch:
            break

        epoch += 1


if __name__ == "__main__":
    main()