import cv2
import torch
import numpy as np
import pandas as pd
import os
from torch.utils.data import Dataset
from facenet_pytorch import MTCNN
import cv2
import torch
import numpy as np
from facenet_pytorch import MTCNN
import os
from contextlib import contextmanager
import pandas as pd
import sys
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import mediapipe as mp
from tqdm import tqdm
import random


@contextmanager
def suppress_c_libs_stderr():
    original_stderr_fd = os.dup(sys.stderr.fileno())
    devnull_fd = os.open(os.os.devnull, os.O_WRONLY)
    os.dup2(devnull_fd, sys.stderr.fileno())
    os.close(devnull_fd)
    try:
        yield
    finally:
        os.dup2(original_stderr_fd, sys.stderr.fileno())
        os.close(original_stderr_fd)


class CASME2_Flow_Dataset(Dataset):

    def __init__(self, csv_path, flow_features_root, data_root,
                 transform=None, mode='train', loso_subject=None,
                 flow_features_root_2=None,
                 use_facial_regions=False,
                 original_image_root='/media/data/lzq/casme2_256',
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

        print(f"CASME2数据集加载成功，模式: {mode}, 有效样本数: {len(self.df)}, 标签列: '{self.label_col}'")

    def __len__(self):
        return len(self.df)

    def get_cls_num_list(self):
        return self.cls_num_list

    def _get_padded_square_bbox(self, points, img_w, img_h, padding_ratio=0.2):
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        width, height = max_x - min_x, max_y - min_y
        center_x, center_y = min_x + width / 2, min_y + height / 2
        side_len = max(width, height) * (1 + padding_ratio)
        half_side = side_len / 2
        left = max(0, int(center_x - half_side))
        top = max(0, int(center_y - half_side))
        right = min(img_w - 1, int(center_x + half_side))
        bottom = min(img_h - 1, int(center_y + half_side))
        return (left, top, right, bottom)

    def _get_facial_regions_image(self, flow_image, subject, filename, apex_frame):
        print("警告: `_get_facial_regions_image` 方法在当前 `use_facial_regions` 配置下未被使用。")
        return flow_image

    def _apply_flow_distortion(self, pil_image, alpha, beta):
        img_np = np.array(pil_image).astype(np.float32)
        dx_channel, dy_channel, strain_channel = img_np[:, :, 0], img_np[:, :, 1], img_np[:, :, 2]
        dx_transformed = alpha * dx_channel
        dy_transformed = beta * dy_channel
        transformed_img_np = np.stack([dx_transformed, dy_transformed, strain_channel], axis=-1)
        transformed_img_np = np.clip(transformed_img_np, 0, 255).astype(np.uint8)
        return Image.fromarray(transformed_img_np, 'RGB')

    def __getitem__(self, index):
        row = self.df.iloc[index]
        subject = str(row['Subject']).zfill(2)
        filename = row['Filename']
        onset_frame = row['OnsetFrame']
        apex_frame = row['ApexFrame']
        relative_path = os.path.join(f"sub{subject}", filename, f"flow_{onset_frame}_{apex_frame}.png")

        label = row[self.label_col]

        new_label = self.new_labels[index] if self.new_labels and index < len(self.new_labels) else -1
        try:
            min_val, max_val = self.flow_distortion_range
            alpha, beta = random.uniform(min_val, max_val), random.uniform(min_val, max_val)
            if self.flow_features_root_2:
                path1 = os.path.join(self.flow_features_root, relative_path)
                path2 = os.path.join(self.flow_features_root_2, relative_path)
                image_1, image_2 = Image.open(path1).convert('RGB'), Image.open(path2).convert('RGB')
                if self.use_facial_regions:
                    image_1 = self._apply_flow_distortion(image_1, alpha, 1.0)
                    image_2 = self._apply_flow_distortion(image_2, 1.0, beta)
                if self.transform:
                    return (self.transform(image_1), self.transform(image_2)), label, new_label
                return (image_1, image_2), label, new_label
            else:
                path1 = os.path.join(self.flow_features_root, relative_path)
                image_1 = Image.open(path1).convert('RGB')
                if self.use_facial_regions:
                    image_1 = self._apply_flow_distortion(image_1, alpha, 1.0)
                if self.transform:
                    return self.transform(image_1), label, new_label
                return image_1, label, new_label
        except FileNotFoundError:
            return None


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


class SMIC_Flow_Dataset(Dataset):
    def __init__(self, csv_path, flow_features_root, data_root,
                 transform=None, mode='train', loso_subject=None,
                 flow_features_root_2=None,
                 use_facial_regions=False,
                 original_image_root='/home/lzq/smic_256',
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
            df = df[df[self.label_col].isin(valid_labels)].reset_index(drop=True)

        if loso_subject is not None:
            loso_subject_int = int(loso_subject)
            if 'subject' in df.columns:
                df['subject'] = df['subject'].astype(int)
                if mode == 'train':
                    self.df = df[df['subject'] != loso_subject_int].reset_index(drop=True)
                elif mode == 'test':
                    self.df = df[df['subject'] == loso_subject_int].reset_index(drop=True)
            else:
                self.df = df if mode == 'train' else pd.DataFrame()
        else:
            if mode == 'train':
                self.df = df
            else:
                self.df = pd.DataFrame()

        if not self.df.empty:
            self.cls_num_list = self.df[self.label_col].value_counts().sort_index().tolist()
        else:
            self.cls_num_list = []
        self.new_labels = []
        self.label_map = {0: 'negative', 1: 'positive', 2: 'surprise'}
        self.use_facial_regions = use_facial_regions
        self.flow_distortion_range = flow_distortion_range
        if self.use_facial_regions:
            print(f"自适应光流通道调整功能已启用，缩放范围: {flow_distortion_range}")
        print(f"SMIC数据集加载成功，模式: {mode}, 有效样本数: {len(self.df)}, 标签列: '{self.label_col}'")

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

    def _get_facial_regions_image(self, flow_image, label_int, clip_name, apex_frame):
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
        subject, clip_name = str(row['subject']), row['clip']
        onset_frame, apex_frame = row['onset_frame'], row['apex_frame']
        label = row[self.label_col]
        new_label = self.new_labels[index] if self.new_labels and index < len(self.new_labels) else -1
        relative_path = os.path.join(subject, clip_name, f"flow_{onset_frame}_{apex_frame}.png")
        try:
            min_val, max_val = self.flow_distortion_range
            alpha, beta = random.uniform(min_val, max_val), random.uniform(min_val, max_val)
            if self.flow_features_root_2:
                path1, path2 = os.path.join(self.flow_features_root, relative_path), os.path.join(
                    self.flow_features_root_2, relative_path)
                image_1, image_2 = Image.open(path1).convert('RGB'), Image.open(path2).convert('RGB')
                if self.use_facial_regions:
                    image_1, image_2 = self._apply_flow_distortion(image_1, alpha, 1.0), self._apply_flow_distortion(
                        image_2, 1.0, beta)
                if self.transform:
                    return (self.transform(image_1), self.transform(image_2)), label, new_label
                return (image_1, image_2), label, new_label
            else:
                path1 = os.path.join(self.flow_features_root, relative_path)
                image_1 = Image.open(path1).convert('RGB')
                if self.use_facial_regions:
                    image_1 = self._apply_flow_distortion(image_1, alpha, 1.0)
                if self.transform:
                    return self.transform(image_1), label, new_label
                return image_1, label, new_label
        except FileNotFoundError:
            return None


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

def get_landmarks(image, mtcnn, device):
    pass


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


class FusionMixin:

    def init_fusion_components(self, **kwargs):
        # 1. 保存 RGB 根目录
        if 'original_image_root' in kwargs:
            self.original_image_root = kwargs['original_image_root']
        else:
            self.original_image_root = kwargs.get('data_root', '')

        # 2. 初始化 MTCNN
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mtcnn = MTCNN(margin=0, image_size=28, select_largest=True, post_process=False, device=self.device)

        # 3. 预设 Fallback 关键点
        self.fallback_landmarks = np.array([
            [9, 11], [21, 10], [15, 17], [10, 22], [20, 22]
        ])

    def _process_fusion_data(self, flow_img, rgb_onset, rgb_apex):
        if flow_img is None or rgb_onset is None or rgb_apex is None:
            return None

        x2_img = cv2.resize(rgb_onset, (224, 224))
        x3_img = cv2.resize(rgb_apex, (224, 224))
        x2 = torch.tensor(x2_img).float().permute(2, 0, 1)
        x3 = torch.tensor(x3_img).float().permute(2, 0, 1)

        face_apex_small = cv2.resize(rgb_apex, (28, 28), interpolation=cv2.INTER_AREA)
        try:
            _, _, landmarks = self.mtcnn.detect(face_apex_small, landmarks=True)
            lm = landmarks[0].astype(int) if landmarks is not None else self.fallback_landmarks
        except:
            lm = self.fallback_landmarks

        lm = np.clip(lm, 7, 21)  # 限制范围

        flow_small = cv2.resize(flow_img, (28, 28), interpolation=cv2.INTER_AREA)

        padded_flow = cv2.copyMakeBorder(flow_small, 7, 7, 7, 7, cv2.BORDER_CONSTANT, value=0)

        parts = []
        for i in range(5):
            cx, cy = lm[i][0] + 7, lm[i][1] + 7
            parts.append(padded_flow[cy - 7:cy + 7, cx - 7:cx + 7])


        if len(parts) == 5:
            l_col = cv2.hconcat([parts[0], parts[1]])
            r_col = cv2.hconcat([parts[3], parts[4]])
            final_flow = cv2.vconcat([l_col, r_col])
            x1 = torch.tensor(final_flow).float().permute(2, 0, 1)
            return x1, x2, x3

        return None


# --- CASME II ---
class Fusion_CASME2_Dataset(CASME2_Flow_Dataset, FusionMixin):
    def __init__(self, *args, **kwargs):
        CASME2_Flow_Dataset.__init__(self, *args, **kwargs)
        self.init_fusion_components(**kwargs)
        self.all_x1, self.all_x2, self.all_x3, self.all_y = [], [], [], []
        print("正在将 CASME2 数据预加载至内存")

        for index in tqdm(range(len(self.df))):
            row = self.df.iloc[index]
            subject = str(row['Subject']).zfill(2)
            filename = row['Filename']
            onset, apex = row['OnsetFrame'], row['ApexFrame']
            label = int(row[self.label_col])

            flow_path = os.path.join(self.flow_features_root, f"sub{subject}", filename, f"flow_{onset}_{apex}.png")
            path_o = os.path.join(self.original_image_root, f"sub{subject}", filename, f"img{onset}.jpg")
            path_a = os.path.join(self.original_image_root, f"sub{subject}", filename, f"img{apex}.jpg")

            res = self._process_fusion_data(cv2.imread(flow_path), cv2.imread(path_o), cv2.imread(path_a))
            if res:
                self.all_x1.append(res[0])
                self.all_x2.append(res[1])
                self.all_x3.append(res[2])
                self.all_y.append(label)

    def __len__(self):
        return len(self.all_y)

    def __getitem__(self, index):
        res = (self.all_x1[index], self.all_x2[index], self.all_x3[index])
        return res, self.all_y[index], -1


# --- SAMM ---
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


# --- SMIC ---
class Fusion_SMIC_Dataset(SMIC_Flow_Dataset, FusionMixin):
    def __init__(self, *args, **kwargs):
        SMIC_Flow_Dataset.__init__(self, *args, **kwargs)
        self.init_fusion_components(**kwargs)
        self.all_x1, self.all_x2, self.all_x3, self.all_y = [], [], [], []
        print("正在将 SMIC 数据预加载至内存")

        for index in tqdm(range(len(self.df))):
            row = self.df.iloc[index]
            subject = str(row['subject'])
            clip = str(row['clip'])
            onset, apex = row['onset_frame'], row['apex_frame']
            label = int(row[self.label_col])

            flow_path = os.path.join(self.flow_features_root, subject, clip, f"flow_{onset}_{apex}.png")
            rgb_base_dir = os.path.join(self.original_image_root, subject, clip)
            path_o = self._find_reg_image_file(rgb_base_dir, onset)
            path_a = self._find_reg_image_file(rgb_base_dir, apex)

            if not os.path.exists(flow_path) or not path_o or not os.path.exists(
                    path_o) or not path_a or not os.path.exists(path_a):
                continue

            flow_img = cv2.imread(flow_path)
            onset_img = cv2.imread(path_o)
            apex_img = cv2.imread(path_a)

            if flow_img is None or onset_img is None or apex_img is None:
                continue

            res = self._process_fusion_data(flow_img, onset_img, apex_img)
            if res:
                self.all_x1.append(res[0])
                self.all_x2.append(res[1])
                self.all_x3.append(res[2])
                self.all_y.append(label)

    def _find_reg_image_file(self, base_dir, frame_num):

        possible_filenames = [
            f"reg_image{frame_num}.bmp",
            f"reg_image{frame_num:02d}.bmp",
            f"reg_image{frame_num:03d}.bmp",
            f"reg_image{frame_num:04d}.bmp",
        ]

        for filename in possible_filenames:
            file_path = os.path.join(base_dir, filename)
            if os.path.exists(file_path):
                return file_path

        try:
            bmp_files = [f for f in os.listdir(base_dir)
                         if f.startswith(f"reg_image") and f.endswith(".bmp")]
            for bmp_file in bmp_files:
                num_part = bmp_file.replace("reg_image", "").replace(".bmp", "")
                if num_part.isdigit() and int(num_part) == frame_num:
                    return os.path.join(base_dir, bmp_file)
        except:
            pass

        return None

    def __len__(self):
        return len(self.all_y)

    def __getitem__(self, index):
        res = (self.all_x1[index], self.all_x2[index], self.all_x3[index])
        return res, self.all_y[index], -1