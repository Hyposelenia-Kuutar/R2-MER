import argparse
import csv
import itertools
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs import load_config


TRAIN_SCRIPT = PROJECT_ROOT / "tools" / "train.py"


def parse_args():
    parser = argparse.ArgumentParser("R2-MER LOSO Experiment Launcher")
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "casme2_3class.py"),
        help="Path to a dataset config file.",
    )
    return parser.parse_args()


def get_subject_list(config):
    dataset = config["dataset"]
    data_cfg = config["data"]

    if dataset != "3db":
        df = pd.read_csv(data_cfg["csv_path"])
        col = "Subject" if "Subject" in df.columns else "subject"
        return sorted(df[col].unique().tolist())

    subjects = []
    for dataset_name, dataset_cfg in data_cfg["datasets"].items():
        df = pd.read_csv(dataset_cfg["csv_path"])
        col = "Subject" if "Subject" in df.columns else "subject"
        dataset_subjects = sorted(df[col].unique().tolist())
        subjects.extend([f"{dataset_name}_{subject}" for subject in dataset_subjects])

    return subjects


def run_experiment(config, config_path, params, subjects, experiment_name, global_summary_file):
    print(f"\n{'=' * 60}\n LOSO Experiment: {experiment_name}\n{'=' * 60}")

    grid_result_root = config["output_root"]
    gpu_id = str(config.get("gpu_id", "0"))

    current_exp_dir = os.path.join(grid_result_root, experiment_name)
    preds_dir = os.path.join(current_exp_dir, "predictions")
    weights_dir = os.path.join(current_exp_dir, "weights")
    os.makedirs(preds_dir, exist_ok=True)
    os.makedirs(weights_dir, exist_ok=True)

    detailed_csv = os.path.join(current_exp_dir, f"detailed_{experiment_name}.csv")
    num_classes = params["num_classes"]
    cm_headers = [f"cm_{i}_{j}" for i in range(num_classes) for j in range(num_classes)]

    with open(detailed_csv, mode="w", newline="") as f:
        csv.writer(f).writerow(["Subject", "ACC", "UF1", "UAR"] + cm_headers)

    start_time = time.time()
    all_preds_list, all_targets_list = [], []

    for subject_id in subjects:
        print(f"--- Processing Subject {subject_id} ---")
        safe_subject_name = str(subject_id).replace("/", "_")
        preds_save_path = os.path.join(preds_dir, f"{safe_subject_name}_preds.pt")

        train_cmd = [
            sys.executable,
            str(TRAIN_SCRIPT),
            "--config",
            str(config_path),
            "--loso_subject",
            str(subject_id),
            "--preds_save_path",
            preds_save_path,
            "--weights_save_dir",
            weights_dir,
            "--num_classes",
            str(num_classes),
            "--batch_size",
            str(params["batch_size"]),
            "--epochs",
            str(params["epochs"]),
            "--lr",
            str(params["learning_rate"]),
            "--weight_decay",
            str(params["weight_decay"]),
            "--seed",
            str(params["seed"]),
            "--gpu",
            "0",
        ]

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_id

        try:
            subprocess.run(train_cmd, env=env, check=True)

            if os.path.exists(preds_save_path):
                data = torch.load(preds_save_path, map_location="cpu")
                s_preds = data["preds"].numpy()
                s_targets = data["targets"].numpy()

                all_preds_list.append(data["preds"])
                all_targets_list.append(data["targets"])

                s_acc = accuracy_score(s_targets, s_preds)
                s_uf1 = f1_score(s_targets, s_preds, average="macro")
                s_uar = recall_score(s_targets, s_preds, average="macro")
                s_cm = confusion_matrix(
                    s_targets,
                    s_preds,
                    labels=list(range(num_classes)),
                ).flatten()

                with open(detailed_csv, mode="a", newline="") as f:
                    csv.writer(f).writerow(
                        [
                            subject_id,
                            f"{s_acc:.4f}",
                            f"{s_uf1:.4f}",
                            f"{s_uar:.4f}",
                        ]
                        + s_cm.tolist()
                    )

        except subprocess.CalledProcessError:
            print(f"Error on subject {subject_id}")

    if not all_preds_list:
        print("No valid prediction files were generated for this experiment.")
        return

    final_preds = torch.cat(all_preds_list).numpy()
    final_targets = torch.cat(all_targets_list).numpy()

    overall_acc = accuracy_score(final_targets, final_preds)
    overall_uf1 = f1_score(final_targets, final_preds, average="macro")
    overall_uar = recall_score(final_targets, final_preds, average="macro")
    overall_cm = confusion_matrix(
        final_targets,
        final_preds,
        labels=list(range(num_classes)),
    ).flatten()

    with open(detailed_csv, mode="a", newline="") as f:
        csv.writer(f).writerow(
            [
                "OVERALL",
                f"{overall_acc:.4f}",
                f"{overall_uf1:.4f}",
                f"{overall_uar:.4f}",
            ]
            + overall_cm.tolist()
        )

    duration = (time.time() - start_time) / 3600

    with open(global_summary_file, mode="a", newline="") as f:
        csv.writer(f).writerow(
            [
                experiment_name,
                params["learning_rate"],
                params["weight_decay"],
                params["batch_size"],
                params["epochs"],
                params["seed"],
                f"{overall_uf1:.4f}",
                f"{overall_uar:.4f}",
                f"{overall_acc:.4f}",
                f"{duration:.2f}h",
                detailed_csv,
            ]
        )


def main():
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    config = load_config(config_path)

    grid_result_root = config["output_root"]
    param_grid = config["param_grid"]
    dataset = config["dataset"]

    os.makedirs(grid_result_root, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_csv = os.path.join(
        grid_result_root,
        f"{dataset}_{param_grid['num_classes'][0]}cls_grid_summary_{timestamp}.csv",
    )

    with open(summary_csv, mode="w", newline="") as f:
        csv.writer(f).writerow(
            [
                "Experiment_ID",
                "LR",
                "WD",
                "BS",
                "EP",
                "Seed",
                "Overall_UF1",
                "Overall_UAR",
                "Overall_ACC",
                "Time",
                "Detail_Path",
            ]
        )

    subjects = get_subject_list(config)

    keys, values = zip(*param_grid.items())
    param_combinations = [
        dict(zip(keys, combination))
        for combination in itertools.product(*values)
    ]

    for params in param_combinations:
        exp_name = (
            f"{dataset}_{params['num_classes']}cls_"
            f"lr{params['learning_rate']}_"
            f"wd{params['weight_decay']}_"
            f"bs{params['batch_size']}_"
            f"ep{params['epochs']}_"
            f"seed{params['seed']}"
        )

        run_experiment(
            config=config,
            config_path=config_path,
            params=params,
            subjects=subjects,
            experiment_name=exp_name,
            global_summary_file=summary_csv,
        )


if __name__ == "__main__":
    main()