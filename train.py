import os
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
import random
from dataset import Fusion_CASME2_Dataset, Fusion_SAMM_Dataset, Fusion_SMIC_Dataset
from Model import FusionHTNet
from torch.utils.data import DataLoader, ConcatDataset
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.utils.checkpoint")
warnings.filterwarnings("ignore", message="None of the inputs have requires_grad=True")
def parse_option():
    parser = argparse.ArgumentParser('FusionHTNet Training')
    parser.add_argument('--dataset', type=str, required=True, choices=['casme2', 'samm', 'smic', '3db'], help='选择数据集')
    parser.add_argument('--loso_subject', type=str, required=True)
    parser.add_argument('--preds_save_path', type=str, required=True, help='预测结果保存路径 (.pt)')

    # 权重保存目录
    parser.add_argument('--weights_save_dir', type=str, default='./weights', help='模型权重保存目录')

    # 路径参数
    parser.add_argument('--csv_path', type=str, default='/home/project/lizhiqi/thin/5casme2.csv')
    parser.add_argument('--data_root', type=str, default='/media/data/lizhiqi/casme2_256', help='RGB图片根目录')
    parser.add_argument('--flow_root', type=str,
                        default='/home/project/lizhiqi/thin/CASME2_TVL1/unfiltered/flow_strain', help='光流图片根目录')
    parser.add_argument('--recovery_weights', type=str, default='', help='RegionRecovery 预训练权重')

    # 训练参数
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=250)
    parser.add_argument('--lr', type=float, default=0.00005)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--num_classes', type=int, default=3)

    parser.add_argument('--train_ratio', type=float, default=1.0, help='训练集使用的样本比例 (0.0 - 1.0)')
    parser.add_argument('--seed', type=int, default=42, help='全局随机种子')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='优化器的权重衰减(L2正则化)')

    return parser.parse_args()


def collate_fn_safe(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if not batch: return torch.tensor([]), torch.tensor([]), torch.tensor([])
    return torch.utils.data.dataloader.default_collate(batch)


# 设置随机种子
def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    args = parse_option()
    set_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # 确保权重目录存在
    if not os.path.exists(args.weights_save_dir):
        os.makedirs(args.weights_save_dir)

    # 构造权重保存路径
    weight_path = os.path.join(args.weights_save_dir, f"{args.loso_subject}.pth")

    print(f"--- Running LOSO for Subject {args.loso_subject} ---")
    print(f"--- Weights will be saved to {weight_path} based on Best UF1 ---")

    # --- 数据集选择逻辑 ---
    DatasetClass = None
    if args.dataset == '3db':
        # 解析受试者，格式如 'casme2_01'
        loso_db, sub_id = args.loso_subject.split('_', 1)
        # 预设三个库的基础路径 (与你的环境一致)
        base = '/root/lizhiqi'
        c_csv, c_rgb, c_flow = f'{base}/3casme2.csv', f'{base}/casme2_256', f'{base}/CASME2_TVL1/unfiltered/flow_strain'
        s_csv, s_rgb, s_flow = f'{base}/3samm.csv', f'{base}/samm_256', f'{base}/SAMM_TVL1/unfiltered/flow_strain'
        sm_csv, sm_rgb, sm_flow = f'{base}/3smic.csv', f'{base}/smic_256', f'{base}/SMIC_TVL1/unfiltered/flow_strain'

        # ⭐️ 动态加载并拼接三个数据集作为训练集
        train_list = []
        train_list.append(
            Fusion_CASME2_Dataset(csv_path=c_csv, flow_features_root=c_flow, data_root=c_rgb, original_image_root=c_rgb,
                                  mode='train', loso_subject=sub_id if loso_db == 'casme2' else None,
                                  num_classes=args.num_classes))
        train_list.append(
            Fusion_SAMM_Dataset(csv_path=s_csv, flow_features_root=s_flow, data_root=s_rgb, original_image_root=s_rgb,
                                mode='train', loso_subject=sub_id if loso_db == 'samm' else None,
                                num_classes=args.num_classes))
        train_list.append(Fusion_SMIC_Dataset(csv_path=sm_csv, flow_features_root=sm_flow, data_root=sm_rgb,
                                              original_image_root=sm_rgb, mode='train',
                                              loso_subject=sub_id if loso_db == 'smic' else None,
                                              num_classes=args.num_classes))
        train_ds = ConcatDataset(train_list)

        if loso_db == 'casme2':
            val_ds = Fusion_CASME2_Dataset(csv_path=c_csv, flow_features_root=c_flow, data_root=c_rgb,
                                           original_image_root=c_rgb, mode='test', loso_subject=sub_id,
                                           num_classes=args.num_classes)
        elif loso_db == 'samm':
            val_ds = Fusion_SAMM_Dataset(csv_path=s_csv, flow_features_root=s_flow, data_root=s_rgb,
                                         original_image_root=s_rgb, mode='test', loso_subject=sub_id,
                                         num_classes=args.num_classes)
        elif loso_db == 'smic':
            val_ds = Fusion_SMIC_Dataset(csv_path=sm_csv, flow_features_root=sm_flow, data_root=sm_rgb,
                                         original_image_root=sm_rgb, mode='test', loso_subject=sub_id,
                                         num_classes=args.num_classes)

    else:
        DatasetClass = None
        if args.dataset == 'casme2':
            DatasetClass = Fusion_CASME2_Dataset
        elif args.dataset == 'samm':
            DatasetClass = Fusion_SAMM_Dataset
        elif args.dataset == 'smic':
            DatasetClass = Fusion_SMIC_Dataset

        train_ds = DatasetClass(csv_path=args.csv_path, flow_features_root=args.flow_root, data_root=args.data_root,
                                original_image_root=args.data_root, mode='train', loso_subject=args.loso_subject,
                                num_classes=args.num_classes)
        val_ds = DatasetClass(csv_path=args.csv_path, flow_features_root=args.flow_root, data_root=args.data_root,
                              original_image_root=args.data_root, mode='test', loso_subject=args.loso_subject,
                              num_classes=args.num_classes)
    if args.train_ratio < 1.0:
        print(f"正在进行训练集下采样，保留比例: {args.train_ratio * 100}%")

        # 获取训练集索引
        indices = list(range(len(train_ds)))

        # 提取标签用于分层
        if hasattr(train_ds, 'labels'):
            y_train = train_ds.labels
        else:
            print("正在提取标签以进行分层采样...")
            y_train = [train_ds[i][1] for i in indices]

        if len(indices) > 0:
            try:
                # 使用分层采样 (Stratify)
                train_idx, _ = train_test_split(
                    indices,
                    train_size=args.train_ratio,
                    stratify=y_train,
                    random_state=args.seed
                )
            except ValueError:
                print("样本数过少或类别分布极端，无法进行分层采样，切换为随机采样。")
                train_idx, _ = train_test_split(
                    indices,
                    train_size=args.train_ratio,
                    random_state=args.seed
                )

            train_ds = Subset(train_ds, train_idx)
            print(f"采样后训练集数量: {len(train_ds)}")



    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(args.seed)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0,
                              collate_fn=collate_fn_safe, worker_init_fn=seed_worker, generator=g)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0,
                            collate_fn=collate_fn_safe, worker_init_fn=seed_worker, generator=g)

    # 2. 初始化模型
    htnet_config = {
        'image_size': 28, 'patch_size': 7, 'num_classes': args.num_classes,
        'dim': 256, 'heads': 3, 'num_hierarchies': 3, 'block_repeats': (2, 2, 8)
    }
    model = FusionHTNet(htnet_config=htnet_config, new_module_path=args.recovery_weights)
    model = model.to(device)

    # --- 冻结参数逻辑 ---
    if hasattr(model, 'region_recovery'):
        print("Freezing region_recovery parameters...")
        for param in model.region_recovery.parameters():
            param.requires_grad = False

    if hasattr(model, 'htnet'):
        print("Initial freezing of HTNet parameters (will unfreeze at epoch 100)...")
        for param in model.htnet.parameters():
            param.requires_grad = False

    # 3. 优化器
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_uf1 = 0.0

    best_acc_at_best_uf1 = 0.0

    subject_start_time = time.time()
    epoch = 1
    total_epochs = args.epochs

    # 4. 训练循环
    while epoch <= total_epochs:
        epoch_start_time = time.time()

        if epoch == 100:
            print(f"Epoch {epoch}: Unfreezing HTNet parameters...")
            unfreeze_params = []
            for name, param in model.named_parameters():
                if "htnet" in name and not param.requires_grad:
                    param.requires_grad = True
                    unfreeze_params.append(param)

            if unfreeze_params:
                optimizer.add_param_group({'params': unfreeze_params, 'lr': args.lr})
                print(f"Added {len(unfreeze_params)} params to optimizer.")

        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_data in train_loader:
            if len(batch_data) == 0: continue
            (inputs, label, _) = batch_data
            x1, x2, x3 = inputs

            x1, x2, x3, label = x1.to(device), x2.to(device), x3.to(device), label.to(device)

            optimizer.zero_grad()
            outputs = model(x1, x2, x3)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x1.size(0)
            _, pred = torch.max(outputs, 1)
            train_correct += (pred == label).sum().item()
            train_total += label.size(0)

        # 计算训练指标
        train_avg_loss = train_loss / len(train_loader.dataset) if len(train_loader.dataset) > 0 else 0
        train_acc = train_correct / train_total if train_total > 0 else 0

        print(
            f"Sub {args.loso_subject} | Epoch {epoch}/{total_epochs} | Train Acc: {train_acc:.4f} | Loss: {train_avg_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

        # 5. 验证
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0
        all_preds, all_targets = [], []

        with torch.no_grad():
            for batch_data in val_loader:
                if len(batch_data) == 0: continue
                (inputs, label, _) = batch_data
                x1, x2, x3 = inputs
                x1, x2, x3, label = x1.to(device), x2.to(device), x3.to(device), label.to(device)

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

            val_uf1 = f1_score(all_targets, all_preds, average='macro')

            print(
                f"Sub {args.loso_subject} | Epoch {epoch}/{total_epochs} | Val Acc: {val_acc:.4f} | Val UF1: {val_uf1:.4f} | Val Loss: {val_avg_loss:.4f}")

            if val_uf1 > best_uf1:

                best_uf1 = val_uf1
                best_acc_at_best_uf1 = val_acc

                save_dir = os.path.dirname(args.preds_save_path)
                if not os.path.exists(save_dir): os.makedirs(save_dir)
                torch.save({'preds': torch.tensor(all_preds), 'targets': torch.tensor(all_targets)},
                           args.preds_save_path)

                #torch.save(model.state_dict(), weight_path)
                # print(f"Saved new best model (UF1: {best_uf1:.4f}) to {weight_path}")

                if val_uf1 == 1.0:
                    break

        epoch_duration = time.time() - epoch_start_time
        elapsed_time = time.time() - subject_start_time
        avg_time = elapsed_time / epoch if epoch > 0 else 0
        remaining = (total_epochs - epoch) * avg_time
        print(f"Time: {epoch_duration:.2f}s | Remaining: {remaining:.2f}s\n")


        if train_acc >= 0.9999 and 150 < epoch:
            break
        else:
            epoch += 1


if __name__ == '__main__':
    main()