import os
import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN

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

        # [0, 255] -> [-1, 1]
        x2 = x2 / 127.5 - 1.0
        x3 = x3 / 127.5 - 1.0

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