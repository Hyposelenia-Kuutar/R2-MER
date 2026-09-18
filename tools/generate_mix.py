import os
import cv2
import torch
import numpy as np
import pandas as pd
from facenet_pytorch import MTCNN
from tqdm import tqdm


class RegionSwapper:
    def __init__(self, dataset_configs, save_base_dir, num_classes=3, loso_subject=None, patch_size=56):
        # dataset_configs 格式: [{'name': 'casme2', 'data_root': '...', 'csv_path': '...'}, ...]
        self.dataset_configs = dataset_configs
        self.num_classes = num_classes
        self.loso_subject = str(loso_subject) if loso_subject else None
        self.patch_size = patch_size
        self.half_p = patch_size // 2

        dataset_tag = "_".join(
            sorted([config['name'] for config in self.dataset_configs])
        )

        self.save_dir = os.path.join(
            save_base_dir,
            f"{dataset_tag}_{self.num_classes}",
            f"loso_{self.loso_subject}"
        )
        os.makedirs(self.save_dir, exist_ok=True)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mtcnn = MTCNN(margin=0, image_size=224, select_largest=True, post_process=False, device=self.device)
        self.soft_mask = self._create_soft_mask(patch_size, feather=12)

    def _create_soft_mask(self, size, feather):
        mask = np.zeros((size, size), dtype=np.float32)
        cv2.rectangle(mask, (feather, feather), (size - feather, size - feather), 1.0, -1)
        mask = cv2.GaussianBlur(mask, (feather * 2 + 1, feather * 2 + 1), 0)
        return np.stack([mask] * 3, axis=-1)

    def _find_samm_image(self, root, subject, filename, frame_idx):
        candidates = [f"{subject}_{str(frame_idx).zfill(4)}.jpg", f"{subject}_{str(frame_idx).zfill(5)}.jpg",
                      f"{subject}_{frame_idx}.jpg"]
        for cand in candidates:
            path = os.path.join(root, subject, filename, cand)
            if os.path.exists(path): return path
        return None

    def _find_smic_image(self, base_dir, frame_num):
        f_str = str(frame_num)
        possible_filenames = [
            f"reg_image{f_str}.bmp",
            f"reg_image{f_str.zfill(2)}.bmp",
            f"reg_image{f_str.zfill(3)}.bmp",
            f"reg_image{f_str.zfill(4)}.bmp"
        ]
        for filename in possible_filenames:
            path = os.path.join(base_dir, filename)
            if os.path.exists(path): return path

        if os.path.exists(base_dir):
            bmp_files = [f for f in os.listdir(base_dir) if f.startswith("reg_image") and f.endswith(".bmp")]
            for bmp_file in bmp_files:
                num_part = bmp_file.replace("reg_image", "").replace(".bmp", "")
                if num_part.isdigit() and int(num_part) == int(frame_num):
                    return os.path.join(base_dir, bmp_file)
        return None

    def _get_valid_apex_images(self):
        valid_images_info = []

        # 遍历所有传入的数据集配置
        for config in self.dataset_configs:
            dataset_name = config['name']
            data_root = config['data_root']
            csv_path = config['csv_path']

            if csv_path.endswith('.xlsx') or csv_path.endswith('.xls'):
                df = pd.read_excel(csv_path)
            else:
                df = pd.read_csv(csv_path)

            df.columns = df.columns.str.strip()

            for index, row in df.iterrows():
                subject = None
                if 'Subject' in df.columns:
                    subject = str(row['Subject'])
                elif 'subject' in df.columns:
                    subject = str(row['subject'])

                if self.loso_subject and self.loso_subject.lower() != 'none' and subject is not None:
                    try:
                        if str(int(subject)) == str(int(self.loso_subject)): continue
                    except ValueError:
                        if str(subject) == str(self.loso_subject): continue

                apex_path = None
                if dataset_name == 'casme2':
                    filename = str(row['Filename'])
                    apex = str(row['ApexFrame'])
                    apex_path = os.path.join(data_root, f"sub{subject.zfill(2)}", filename, f"img{apex}.jpg")
                elif dataset_name == 'samm':
                    filename = str(row['Filename'])
                    apex = str(row['Apex Frame'])
                    apex_path = self._find_samm_image(data_root, subject.zfill(3), filename, apex)
                elif dataset_name == 'smic':
                    clip = str(row['clip'])
                    apex = str(row['apex_frame'])
                    base_dir = os.path.join(data_root, subject, clip)
                    apex_path = self._find_smic_image(base_dir, apex)
                elif dataset_name == 'mmew':
                    emotion = str(row.get('Estimated Emotion', '')).strip()
                    filename = str(row.get('Filename', '')).strip()
                    apex = str(row.get('ApexFrame', '')).strip()
                    candidates = [f"{apex}.jpg", f"{apex.zfill(2)}.jpg", f"{apex.zfill(3)}.jpg", f"{apex.zfill(4)}.jpg", f"{apex.zfill(5)}.jpg"]
                    for cand in candidates:
                        temp_path = os.path.join(data_root, emotion, filename, cand)
                        if os.path.exists(temp_path):
                            apex_path = temp_path
                            break

                if apex_path and os.path.exists(apex_path):
                    valid_images_info.append({
                        'path': apex_path,
                        'dataset': dataset_name,
                        'root': data_root
                    })

        return valid_images_info

    def generate(self):
        valid_images = self._get_valid_apex_images()
        N = len(valid_images)
        if N == 0:
            print(f"未找到Apex 帧。")
            return self.save_dir

        print(f"提取了来自 {len(self.dataset_configs)} 个数据集的 {N} 张 Apex 帧 (已剔除受试者 {self.loso_subject})。保存至: {self.save_dir}")
        cache_data = []

        for item in tqdm(valid_images, desc="Caching Landmarks"):
            img_path = item['path']
            img = cv2.imread(img_path)
            if img is None: continue
            img_resized = cv2.resize(img, (224, 224))
            try:
                _, _, lmk = self.mtcnn.detect(img_resized, landmarks=True)
                if lmk is not None:
                    safe_lmk = np.clip(lmk[0].astype(int), self.half_p, 224 - self.half_p - 1)
                    cache_data.append({
                        'img': img_resized,
                        'lmk': safe_lmk,
                        'name': os.path.basename(img_path),
                        'path': img_path,
                        'dataset': item['dataset'],
                        'root': item['root']
                    })
            except:
                continue

        swap_idx = [0, 1, 3, 4]
        for target in tqdm(cache_data, desc="Swapping Apex Regions (Cross-Dataset)"):
            rel_dir = os.path.relpath(os.path.dirname(target['path']), target['root'])
            target_sub_dir = os.path.join(self.save_dir, target['dataset'], rel_dir, target['name'])
            os.makedirs(target_sub_dir, exist_ok=True)

            for source_idx, source in enumerate(cache_data):
                mix_img = target['img'].copy().astype(np.float32)

                for i in swap_idx:
                    tx, ty = target['lmk'][i]
                    sx, sy = source['lmk'][i]

                    s_patch = source['img'][
                              sy - self.half_p:sy + self.half_p,
                              sx - self.half_p:sx + self.half_p
                              ].astype(np.float32)

                    mix_img[
                    ty - self.half_p:ty + self.half_p,
                    tx - self.half_p:tx + self.half_p
                    ] = s_patch

                save_name = (
                    f"mix_{source_idx:04d}_"
                    f"{source['dataset']}_"
                    f"{source['name']}"
                )

                cv2.imwrite(
                    os.path.join(target_sub_dir, save_name),
                    mix_img.astype(np.uint8)
                )
        return self.save_dir


def prepare_pretrain_data(dataset_configs, save_base, num_classes, subject):
    swapper = RegionSwapper(dataset_configs, save_base, num_classes, subject)
    if os.path.exists(swapper.save_dir) and len(os.listdir(swapper.save_dir)) > 0:
        return swapper.save_dir
    return swapper.generate()