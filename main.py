import os
import subprocess
import pandas as pd
import itertools
import csv
import time
import torch
import numpy as np
from datetime import datetime
from sklearn.metrics import f1_score, recall_score, accuracy_score, confusion_matrix

# ================= 配置区域 =================
BASE_ROOT = '/root/lizhiqi'
CSV_FILE_PATH = os.path.join(BASE_ROOT, '3casme2.csv')
DATA_ROOT = os.path.join(BASE_ROOT, 'casme2_256')
FLOW_ROOT = os.path.join(BASE_ROOT, 'CASME2/flow_strain')

SAVE_ROOT = '/root/NFS_data'
GRID_RESULT_ROOT = os.path.join(SAVE_ROOT, 'grid_search_3casme2_')
GPU_ID = '3'


UNIVERSAL_WEIGHT_PATH = os.path.join(SAVE_ROOT, 'pretrain_pool', 'pretrain.pth')

PARAM_GRID = {
    'learning_rate': [0.00005],
    'weight_decay': [0.0001],
    'batch_size': [128],
    'epochs': [500],
    'dataset': ['casme2'],
    'num_classes': [3],
    'seed': [42]
}


# ===========================================

def get_subject_list(csv_path):
    df = pd.read_csv(csv_path)
    col = 'Subject' if 'Subject' in df.columns else 'subject'
    return sorted(df[col].unique().tolist())


def run_experiment(params, subjects, experiment_name, global_summary_file):
    print(f"\n{'=' * 60}\n LOSO 实验: {experiment_name}\n{'=' * 60}")

    current_exp_dir = os.path.join(GRID_RESULT_ROOT, experiment_name)
    preds_dir = os.path.join(current_exp_dir, 'predictions')
    weights_dir = os.path.join(current_exp_dir, 'weights')
    os.makedirs(preds_dir, exist_ok=True)
    os.makedirs(weights_dir, exist_ok=True)

    detailed_csv = os.path.join(current_exp_dir, f'detailed_{experiment_name}.csv')
    num_classes = params['num_classes']
    cm_headers = [f'cm_{i}_{j}' for i in range(num_classes) for j in range(num_classes)]
    with open(detailed_csv, mode='w', newline='') as f:
        csv.writer(f).writerow(['Subject', 'ACC', 'UF1', 'UAR'] + cm_headers)

    start_time = time.time()
    all_preds_list, all_targets_list = [], []

    for subject_id in subjects:
        print(f"--- Processing Subject {subject_id} ---")
        preds_save_path = os.path.join(preds_dir, f'{subject_id}_preds.pt')


        train_cmd = (
            f"CUDA_VISIBLE_DEVICES={GPU_ID} python train.py "
            f"--dataset {params['dataset']} "
            f"--loso_subject {subject_id} "
            f"--preds_save_path {preds_save_path} "
            f"--weights_save_dir {weights_dir} "
            f"--csv_path {CSV_FILE_PATH} "
            f"--data_root {DATA_ROOT} "
            f"--flow_root {FLOW_ROOT} "
            f"--recovery_weights {UNIVERSAL_WEIGHT_PATH} " 
            f"--num_classes {num_classes} "
            f"--batch_size {params['batch_size']} "
            f"--epochs {params['epochs']} "
            f"--lr {params['learning_rate']} "
            f"--weight_decay {params['weight_decay']} "
            f"--gpu 0 "
            f"--seed {params['seed']}"
        )

        try:
            subprocess.run(train_cmd, shell=True, check=True)
            if os.path.exists(preds_save_path):
                data = torch.load(preds_save_path)
                s_preds, s_targets = data['preds'].numpy(), data['targets'].numpy()
                all_preds_list.append(data['preds'])
                all_targets_list.append(data['targets'])

                s_acc = accuracy_score(s_targets, s_preds)
                s_uf1 = f1_score(s_targets, s_preds, average='macro')
                s_uar = recall_score(s_targets, s_preds, average='macro')
                s_cm = confusion_matrix(s_targets, s_preds, labels=list(range(num_classes))).flatten()
                with open(detailed_csv, mode='a', newline='') as f:
                    csv.writer(f).writerow([subject_id, f"{s_acc:.4f}", f"{s_uf1:.4f}", f"{s_uar:.4f}"] + s_cm.tolist())
        except subprocess.CalledProcessError:
            print(f" Error on subject {subject_id}")

    if all_preds_list:
        final_preds, final_targets = torch.cat(all_preds_list).numpy(), torch.cat(all_targets_list).numpy()
        overall_acc = accuracy_score(final_targets, final_preds)
        overall_uf1 = f1_score(final_targets, final_preds, average='macro')
        overall_uar = recall_score(final_targets, final_preds, average='macro')
        overall_cm = confusion_matrix(final_targets, final_preds, labels=list(range(num_classes))).flatten()
        with open(detailed_csv, mode='a', newline='') as f:
            csv.writer(f).writerow(
                ['OVERALL', f"{overall_acc:.4f}", f"{overall_uf1:.4f}", f"{overall_uar:.4f}"] + overall_cm.tolist())

    duration = (time.time() - start_time) / 3600
    with open(global_summary_file, mode='a', newline='') as f:
        csv.writer(f).writerow([
            experiment_name, params['learning_rate'], params['weight_decay'], params['batch_size'], params['epochs'], params['seed'],
            f"{overall_uf1:.4f}", f"{overall_uar:.4f}", f"{overall_acc:.4f}", f"{duration:.2f}h", detailed_csv
        ])


def main():
    if not os.path.exists(GRID_RESULT_ROOT): os.makedirs(GRID_RESULT_ROOT)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    summary_csv = os.path.join(GRID_RESULT_ROOT,
                               f"{PARAM_GRID['dataset'][0]}_{PARAM_GRID['num_classes'][0]}cls_grid_summary_{timestamp}.csv")

    with open(summary_csv, mode='w', newline='') as f:
        csv.writer(f).writerow(
            ['Experiment_ID', 'LR', 'WD', 'BS', 'EP', 'Seed', 'Overall_UF1', 'Overall_UAR', 'Overall_ACC', 'Time',
             'Detail_Path'])

    subjects = get_subject_list(CSV_FILE_PATH)
    keys, values = zip(*PARAM_GRID.items())
    for params in [dict(zip(keys, v)) for v in itertools.product(*values)]:
        exp_name = f"{params['dataset']}_{params['num_classes']}cls_lr{params['learning_rate']}_wd{params['weight_decay']}_bs{params['batch_size']}_ep{params['epochs']}_seed{params['seed']}"

        run_experiment(params, subjects, exp_name, summary_csv)


if __name__ == '__main__':
    main()