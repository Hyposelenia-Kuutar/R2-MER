import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import cv2
from new_model_5_ import RegionRecoveryModel
from generate_mix import prepare_pretrain_data
from pytorch_msssim import SSIM

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'


class MicroExprDataset(Dataset):
    def __init__(self, safe_mix_root, orig_data_roots_dict):
        self.valid_pairs = []

        for root, _, files in os.walk(safe_mix_root):
            for file in files:
                if file.startswith("mix_") and file.lower().endswith(('.jpg', '.png', '.bmp')):
                    mix_path = os.path.join(root, file)


                    rel_path_full = os.path.relpath(root, safe_mix_root)
                    parts = rel_path_full.split(os.sep)

                    if len(parts) < 2:
                        continue

                    dataset_name = parts[0]

                    orig_rel_path = os.path.join(*parts[1:])

                    if dataset_name in orig_data_roots_dict:
                        orig_root = orig_data_roots_dict[dataset_name]
                        orig_path = os.path.join(orig_root, orig_rel_path)

                        if os.path.exists(orig_path):
                            self.valid_pairs.append((orig_path, mix_path))

        print(f"成功装载 {len(self.valid_pairs)} 对图像")

    def __len__(self):
        return len(self.valid_pairs)

    def __getitem__(self, idx):
        orig_path, mix_path = self.valid_pairs[idx]
        return {'mixed': self._load(mix_path), 'target': self._load(orig_path)}

    def _load(self, p):
        img = cv2.imread(p)
        if img is None:
            raise ValueError(f"OpenCV 读取失败: {p}")
        img = cv2.resize(img, (224, 224))
        return torch.tensor(img).float().permute(2, 0, 1) / 255.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', type=str, default=['casme2', 'smic', 'samm'])
    parser.add_argument('--orig_data_roots', nargs='+', type=str,
                        default=['/root/lizhiqi/casme2', '/root/lizhiqi/smic_256', '/root/lizhiqi/samm'])
    parser.add_argument('--csv_paths', nargs='+', type=str,
                        default=['/root/lizhiqi/casme2.csv', '/root/lizhiqi/3smic.csv', '/root/lizhiqi/samm.csv'])

    parser.add_argument('--test_subject', type=str, default='None')
    parser.add_argument('--save_path', type=str, default='/root/NFS_data/pretrain_pool/universal_pretrain.pth')

    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--mix_save_base', type=str, default='/root/NFS_data/multi_exchange_pool')

    parser.add_argument('--epochs', type=int, default=100)

    parser.add_argument('--batch_size', type=int, default=128)

    parser.add_argument('--lambda_l1', type=float, default=0.6)
    parser.add_argument('--lambda_ssim', type=float, default=0.4)

    args = parser.parse_args()

    assert len(args.datasets) == len(args.orig_data_roots) == len(args.csv_paths), "数据集、根路径、CSV的列表长度必须一致"

    dataset_configs = []
    orig_data_roots_dict = {}
    for d_name, d_root, c_path in zip(args.datasets, args.orig_data_roots, args.csv_paths):
        dataset_configs.append({'name': d_name, 'data_root': d_root, 'csv_path': c_path})
        orig_data_roots_dict[d_name] = d_root

    actual_subject = None if args.test_subject.lower() == 'none' else args.test_subject

    safe_mix_folder = prepare_pretrain_data(
        dataset_configs=dataset_configs,
        save_base=args.mix_save_base,
        num_classes=args.num_classes,
        subject=actual_subject
    )

    dataloader = DataLoader(MicroExprDataset(safe_mix_folder, orig_data_roots_dict), batch_size=args.batch_size,
                            shuffle=True)

    if len(dataloader.dataset) == 0:
        print("DataLoader 为空")
        return

    model = RegionRecoveryModel().cuda()

    if torch.cuda.device_count() > 1:
        print(f"检测到 {torch.cuda.device_count()} 张 GPU")
        model = nn.DataParallel(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    criterion_l1 = nn.L1Loss()
    criterion_ssim = SSIM(data_range=1.0, size_average=True, channel=3)

    save_dir = os.path.dirname(args.save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
        print(f"自动创建了缺失的保存目录: {save_dir}")

    for epoch in range(args.epochs):
        model.train()
        l_sum = 0
        for b in dataloader:
            optimizer.zero_grad()

            target_img = b['target'].cuda()
            mixed_img = b['mixed'].cuda()

            out = model(target_img, mixed_img, True)

            loss_l1 = criterion_l1(out, target_img)
            loss_ssim = 1 - criterion_ssim(out, target_img)
            loss = (args.lambda_l1 * loss_l1) + (args.lambda_ssim * loss_ssim)

            if torch.cuda.device_count() > 1:
                loss = loss.mean()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            l_sum += loss.item()

        scheduler.step()

        current_loss = l_sum / len(dataloader)

        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch + 1}/{args.epochs} | LR: {current_lr:.6f} | Total Loss: {current_loss:.4f}")

        final_state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()

        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': final_state_dict,
            'loss': current_loss,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict()
        }, args.save_path)

        print(f"权重已覆盖保存 (最新 Epoch: {epoch + 1})")

    print(f"预训练完成，最终权重已保留至 {args.save_path}")


if __name__ == "__main__":
    main()