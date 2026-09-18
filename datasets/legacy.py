import os

import cv2
import numpy as np
import pandas as pd
import torch
from facenet_pytorch import MTCNN
from torch.utils.data import Dataset

from .casme2 import CASME2_Flow_Dataset
from .samm import SAMM_Flow_Dataset
from .smic import SMIC_Flow_Dataset


class Combined_Dataset(Dataset):
    def __init__(self, transform=None, mode='train', loso_subject=None,
                 casme2_paths: dict = None,
                 samm_paths: dict = None,
                 smic_paths: dict = None,
                 use_facial_regions=False,
                 flow_distortion_range=(0.25, 0.75),
                 num_classes=3
                 ):
        super().__init__()
        casme2_loso, samm_loso, smic_loso = None, None, None
        if loso_subject:
            try:
                dataset_name, subject_id = loso_subject.split('_')
                subject_id = int(subject_id)
                if dataset_name == 'casme2':
                    casme2_loso = subject_id
                elif dataset_name == 'samm':
                    samm_loso = subject_id
                elif dataset_name == 'smic':
                    smic_loso = subject_id
            except (ValueError, IndexError):
                print(f"警告: 无效的 loso_subject 格式 '{loso_subject}'。将被忽略。")
        self.datasets = []
        if casme2_paths and casme2_paths.get('csv_path'):
            self.datasets.append(CASME2_Flow_Dataset(
                csv_path=casme2_paths['csv_path'],
                flow_features_root=casme2_paths['flow_features_root'],
                original_image_root=casme2_paths.get('original_image_root'),
                transform=transform, mode=mode, loso_subject=casme2_loso,
                flow_features_root_2=casme2_paths.get('flow_features_root_2'),
                use_facial_regions=use_facial_regions,
                flow_distortion_range=flow_distortion_range,
                data_root=None,
                num_classes=num_classes
            ))
        if samm_paths and samm_paths.get('csv_path'):
            self.datasets.append(SAMM_Flow_Dataset(
                csv_path=samm_paths['csv_path'],
                flow_features_root=samm_paths['flow_features_root'],
                original_image_root=samm_paths.get('original_image_root'),
                transform=transform, mode=mode, loso_subject=samm_loso,
                flow_features_root_2=samm_paths.get('flow_features_root_2'),
                use_facial_regions=use_facial_regions,
                flow_distortion_range=flow_distortion_range,
                data_root=None,
                num_classes=num_classes
            ))
        if smic_paths and smic_paths.get('csv_path'):
            self.datasets.append(SMIC_Flow_Dataset(
                csv_path=smic_paths['csv_path'],
                flow_features_root=smic_paths['flow_features_root'],
                original_image_root=smic_paths.get('original_image_root'),
                transform=transform, mode=mode, loso_subject=smic_loso,
                flow_features_root_2=smic_paths.get('flow_features_root_2'),
                use_facial_regions=use_facial_regions,
                flow_distortion_range=flow_distortion_range,
                data_root=None,
                num_classes=num_classes
            ))
        self.cumulative_sizes = self.cumsum([len(d) for d in self.datasets])
        print("=" * 50 + f"\n组合数据集加载成功！模式: {mode}, 总样本数: {len(self)}\n" + "=" * 50)

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence: r.append(e + s); s += e
        return r

    def __len__(self):
        return self.cumulative_sizes[-1] if self.cumulative_sizes else 0

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self): raise ValueError("abs(index) > len(dataset)")
            idx = len(self) + idx
        dataset_idx = np.searchsorted(self.cumulative_sizes, idx, side='right')
        sample_idx = idx if dataset_idx == 0 else idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]

    @property
    def new_labels(self):
        return [label for ds in self.datasets for label in ds.new_labels]

    @new_labels.setter
    def new_labels(self, targets: list):
        if not targets: return
        print("=> Combined_Dataset: 正在将新的聚类标签分配给子数据集...")
        start_idx = 0
        for ds in self.datasets:
            end_idx = start_idx + len(ds)
            ds.new_labels = targets[start_idx:end_idx]
            start_idx = end_idx


class FusionHTNet_Dataset(Dataset):
    def __init__(self, csv_path, data_root, mode='train', loso_subject=None,
                 image_size=28, crop_size=14, num_classes=3):
        super().__init__()
        self.data_root = data_root
        self.image_size = image_size
        self.crop_size = crop_size

        df = pd.read_csv(csv_path)

        if loso_subject is not None:
            sub_col = 'Subject' if 'Subject' in df.columns else 'sub'
            df[sub_col] = df[sub_col].astype(str)
            loso_subject = str(loso_subject)
            if mode == 'train':
                self.df = df[df[sub_col] != loso_subject].reset_index(drop=True)
            else:
                self.df = df[df[sub_col] == loso_subject].reset_index(drop=True)
        else:
            self.df = df

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mtcnn = MTCNN(margin=0, image_size=self.image_size, select_largest=True, post_process=False,
                           device=self.device)

        print(f"FusionHTNet 数据集加载: {mode}, 样本数: {len(self.df)}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        subject = str(row['Subject'])
        filename = str(row['Filename'])
        onset_idx = str(row['OnsetFrame'])
        apex_idx = str(row['ApexFrame'])
        label = int(row['label'])

        img_path_apex = os.path.join(self.data_root, subject, filename, f"img{apex_idx}.jpg")
        img_path_onset = os.path.join(self.data_root, subject, filename, f"img{onset_idx}.jpg")

        flow_name = f"flow_{onset_idx}_{apex_idx}.png"  # 假设命名格式
        flow_path = os.path.join(self.data_root, 'optical_flow', subject, filename, flow_name)

        img_apex = cv2.imread(img_path_apex)
        img_onset = cv2.imread(img_path_onset)
        flow_image = cv2.imread(flow_path)

        if img_apex is None or flow_image is None:

            return None

        x2 = cv2.resize(img_onset, (self.image_size, self.image_size))
        x3 = cv2.resize(img_apex, (self.image_size, self.image_size))

        # 转 Tensor [C, H, W]
        x2 = torch.tensor(x2).float().permute(2, 0, 1)
        x3 = torch.tensor(x3).float().permute(2, 0, 1)

        face_apex = cv2.resize(img_apex, (28, 28))
        _, _, landmarks = self.mtcnn.detect(face_apex, landmarks=True)

        if landmarks is None:

            lm = np.array([[9, 11], [21, 10], [15, 17], [10, 22], [20, 22]])
        else:
            lm = landmarks[0].astype(int)

            lm = np.clip(lm, 7, 21)


        five_parts = []
        for i in range(5):
            center_x, center_y = lm[i][0], lm[i][1]

            part = flow_image[center_y - 7:center_y + 7, center_x - 7:center_x + 7]
            five_parts.append(part)

        if len(five_parts) == 5:
            l_part = cv2.hconcat([five_parts[0], five_parts[1]])
            r_part = cv2.hconcat([five_parts[3], five_parts[4]])
            final_flow = cv2.vconcat([l_part, r_part])  # 上下拼

            x1 = torch.tensor(final_flow).float().permute(2, 0, 1)  # [3, H, W]
        else:
            return None
        return (x1, x2, x3), label