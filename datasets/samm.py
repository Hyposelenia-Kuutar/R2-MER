import os
import random

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

from .base import FusionMixin


class SAMM_Flow_Dataset(Dataset):
    def __init__(self, csv_path, flow_features_root, data_root,
                 transform=None, mode='train', loso_subject=None,
                 flow_features_root_2=None,
                 use_facial_regions=False,
                 original_image_root='/media/data/lzq/samm_256',
                 flow_distortion_range=(0.25, 0.75),
                 num_classes=3
                 ):
        super().__init__()
        self.flow_features_root = flow_features_root
        self.flow_features_root_2 = None
        self.transform = transform
        self.num_classes = num_classes
        self.label_col = f'{self.num_classes}label'
        df = pd.read_csv(csv_path)

        df.columns = df.columns.str.strip()

        valid_labels = list(range(self.num_classes))
        if self.label_col in df.columns:
            df[self.label_col] = pd.to_numeric(df[self.label_col], errors='coerce')
            df.dropna(subset=[self.label_col], inplace=True)
            df[self.label_col] = df[self.label_col].astype(int)
            df = df[df[self.label_col].isin(valid_labels)].reset_index(drop=True)

        if loso_subject is not None:
            loso_subject_int = int(loso_subject)
            if 'Subject' in df.columns:
                df['Subject'] = df['Subject'].astype(int)
                if mode == 'train':
                    self.df = df[df['Subject'] != loso_subject_int].reset_index(drop=True)
                elif mode == 'test':
                    self.df = df[df['Subject'] == loso_subject_int].reset_index(drop=True)
            else:
                self.df = df if mode == 'train' else pd.DataFrame()
        else:
            self.df = df if mode == 'train' else pd.DataFrame()

        if not self.df.empty:
            self.cls_num_list = self.df[self.label_col].value_counts().sort_index().tolist()
        else:
            self.cls_num_list = []

        self.new_labels = []
        self.use_facial_regions = use_facial_regions
        self.flow_distortion_range = flow_distortion_range

        if self.use_facial_regions:
            print(f"自适应光流通道调整功能已启用，缩放范围: {flow_distortion_range}")

        print(f"SAMM数据集加载成功，模式: {mode}, 有效样本数: {len(self.df)}, 标签列: '{self.label_col}'")

    def __len__(self):
        return len(self.df)

    def get_cls_num_list(self):
        return self.cls_num_list

    def _get_padded_square_bbox(self, points, img_w, img_h, padding_ratio=0.2):
        x_coords, y_coords = [p[0] for p in points], [p[1] for p in points]
        min_x, max_x, min_y, max_y = min(x_coords), max(x_coords), min(y_coords), max(y_coords)
        width, height = max_x - min_x, max_y - min_y
        center_x, center_y = min_x + width / 2, min_y + height / 2
        side_len = max(width, height) * (1 + padding_ratio)
        half_side = side_len / 2
        left, top = max(0, int(center_x - half_side)), max(0, int(center_y - half_side))
        right, bottom = min(img_w - 1, int(center_x + half_side)), min(img_h - 1, int(center_y + half_side))
        return left, top, right, bottom

    def _get_facial_regions_image(self, flow_image, subject_padded, filename, apex_frame):
        print("警告: `_get_facial_regions_image` 方法在当前 `use_facial_regions` 配置下未被使用。")
        return flow_image

    def _apply_flow_distortion(self, pil_image, alpha, beta):
        img_np = np.array(pil_image).astype(np.float32)
        dx_channel, dy_channel, strain_channel = img_np[:, :, 0], img_np[:, :, 1], img_np[:, :, 2]
        dx_transformed, dy_transformed = alpha * dx_channel, beta * dy_channel
        transformed_img_np = np.stack([dx_transformed, dy_transformed, strain_channel], axis=-1)
        transformed_img_np = np.clip(transformed_img_np, 0, 255).astype(np.uint8)
        return Image.fromarray(transformed_img_np, 'RGB')

    def __getitem__(self, index):
        row = self.df.iloc[index]
        subject = str(row['Subject'])
        filename = row['Filename']
        onset_frame = row['Onset Frame']
        apex_frame = row['Apex Frame']

        subject_padded = subject.zfill(3)
        relative_path = os.path.join(subject_padded, filename, f"flow_{onset_frame}_{apex_frame}.png")

        label = row[self.label_col]
        new_label = self.new_labels[index] if self.new_labels and index < len(self.new_labels) else -1

        try:
            min_val, max_val = self.flow_distortion_range
            alpha = random.uniform(min_val, max_val)
            beta = random.uniform(min_val, max_val)

            if self.flow_features_root_2:
                path1 = os.path.join(self.flow_features_root, relative_path)
                path2 = os.path.join(self.flow_features_root_2, relative_path)
                image_1 = Image.open(path1).convert('RGB')
                image_2 = Image.open(path2).convert('RGB')

                if self.use_facial_regions:
                    image_1 = self._apply_flow_distortion(image_1, alpha, 1.0)
                    image_2 = self._apply_flow_distortion(image_2, 1.0, beta)

                if self.transform:
                    view1 = self.transform(image_1)
                    view2 = self.transform(image_2)
                    return (view1, view2), label, new_label
                else:
                    return (image_1, image_2), label, new_label
            else:
                path1 = os.path.join(self.flow_features_root, relative_path)
                image_1 = Image.open(path1).convert('RGB')

                if self.use_facial_regions:
                    image_1 = self._apply_flow_distortion(image_1, alpha, 1.0)

                if self.transform:
                    view1 = self.transform(image_1)
                    return view1, label, new_label
                else:
                    return image_1, label, new_label

        except FileNotFoundError:
            return None


class Fusion_SAMM_Dataset(SAMM_Flow_Dataset, FusionMixin):
    def __init__(self, *args, **kwargs):
        SAMM_Flow_Dataset.__init__(self, *args, **kwargs)
        self.init_fusion_components(**kwargs)
        self.all_x1, self.all_x2, self.all_x3, self.all_y = [], [], [], []
        print("正在将 SAMM 数据预加载至内存")

        for index in tqdm(range(len(self.df))):
            row = self.df.iloc[index]
            subject = str(row['Subject']).zfill(3)
            filename = str(row['Filename'])
            onset, apex = row['Onset Frame'], row['Apex Frame']
            label = int(row[self.label_col])

            flow_path = os.path.join(self.flow_features_root, subject, filename, f"flow_{onset}_{apex}.png")
            path_o = self._find_image_path(self.original_image_root, subject, filename, onset)
            path_a = self._find_image_path(self.original_image_root, subject, filename, apex)

            res = self._process_fusion_data(cv2.imread(flow_path), cv2.imread(path_o), cv2.imread(path_a))
            if res:
                self.all_x1.append(res[0])
                self.all_x2.append(res[1])
                self.all_x3.append(res[2])
                self.all_y.append(label)

    def _find_image_path(self, root, subject, filename, frame_idx):

        candidates = [
            f"{subject}_{str(frame_idx).zfill(4)}.jpg",
            f"{subject}_{str(frame_idx).zfill(5)}.jpg",
            f"{subject}_{frame_idx}.jpg"
        ]

        for cand in candidates:
            path = os.path.join(root, subject, filename, cand)
            if os.path.exists(path):
                return path

        return os.path.join(root, subject, filename, candidates[1])

    def __len__(self):
        return len(self.all_y)

    def __getitem__(self, index):
        res = (self.all_x1[index], self.all_x2[index], self.all_x3[index])
        return res, self.all_y[index], -1