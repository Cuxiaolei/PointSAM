"""
We follow how torchvision transforms are implemented, and use `set_transform` method to apply data augmentation to the dataset.
It is easier to compose multiple data augmentation methods, and specify them in a config file.
Note that `set_transform` handles a batch of examples. `set_transform` is also compatible with IterableDataset.
Refer to https://huggingface.co/docs/datasets/v2.0.0/en/image_process#data-augmentation.
"""

from typing import Callable, Dict, List
import numpy as np
import torch
from scipy.spatial.transform import Rotation


class Compose:
    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms

    def __call__(self, examples):
        for transform in self.transforms:
            examples = transform(examples)
        return examples

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string


class Transform:
    def apply(self, example: dict):
        return example

    def __call__(self, examples: Dict[str, List]):
        keys = examples.keys()
        ret = [
            self.apply({k: v for k, v in zip(keys, example)})
            for example in zip(*[examples[k] for k in keys])
        ]
        examples.update({k: [x[k] for x in ret] for k in keys})
        return examples


class ToTensor(Transform):
    def apply(self, example):
        # 处理6通道特征和坐标为float张量
        for k in ["coords", "features"]:
            example[k] = torch.tensor(np.array(example[k]), dtype=torch.float)
        # 处理掩码为bool张量
        for k in ["gt_masks"]:
            example[k] = torch.tensor(np.array(example[k]), dtype=torch.bool)
        return example


def normalize_points(points: np.ndarray):
    """Normalize the point cloud into a unit sphere."""
    assert points.ndim == 2 and points.shape[1] == 3, points.shape
    centroid = np.mean(points, axis=0)
    points = points - centroid
    norm = np.max(np.linalg.norm(points, ord=2, axis=1))
    return points / norm


class NormalizePoints(Transform):
    def apply(self, example):
        example["coords"] = normalize_points(np.array(example["coords"]))
        return example


class NormalizeColor(Transform):
    def __init__(self, mean=None, std=None, color_channels=3):
        self.mean = mean  # 应为3元素，对应RGB通道
        self.std = std  # 应为3元素，对应RGB通道
        self.color_channels = color_channels  # 前3通道为颜色

    def apply(self, example):
        features = np.array(example["features"])  # (N, 6)

        # 仅对前3通道颜色进行归一化
        color = features[:, :self.color_channels] / 255.0  # RGB从[0,255]转为[0,1]

        # 应用均值和标准差标准化
        if self.mean is not None:
            color = color - self.mean
        if self.std is not None:
            color = color / self.std

        # 拼接归一化后的颜色和原始法向量
        features[:, :self.color_channels] = color
        example["features"] = features
        return example


class RandomSample(Transform):
    """Randomly sample a fixed number of points from the point cloud."""

    def __init__(self, num_samples: int, replace=False):
        self.num_samples = num_samples
        self.replace = replace

    def apply(self, example):
        coords = np.asarray(example["coords"])
        features = np.asarray(example["features"])  # 6通道特征
        gt_masks = np.array(example["gt_masks"])  # [M, N]

        indices = np.random.choice(len(coords), self.num_samples, replace=self.replace)
        # 如果没有前景，重新采样
        if not (gt_masks[:, indices] == 1).any():
            fg_indices = np.nonzero((gt_masks == 1).any(axis=0))[0]
            bg_indices = np.nonzero((gt_masks == 0).all(axis=0))[0]
            n_fg = int(np.ceil(self.num_samples / len(coords) * len(fg_indices)))
            n_fg = min(n_fg, min(len(fg_indices), self.num_samples))
            fg_indices = np.random.choice(fg_indices, n_fg)
            bg_indices = np.random.choice(bg_indices, self.num_samples - n_fg)
            indices = np.random.permutation(np.concatenate([fg_indices, bg_indices]))

        # 对坐标和6通道特征同时采样
        example["coords"] = coords[indices]
        example["features"] = features[indices]  # 保持6通道结构

        # 处理掩码
        gt_masks = gt_masks[:, indices]
        is_empty_mask = (gt_masks == 0).all(axis=1)
        if is_empty_mask.any():
            gt_masks[is_empty_mask] = gt_masks[~is_empty_mask][0]
        example["gt_masks"] = gt_masks

        return example


# 全局索引生成（根据实际最大点数量调整，确保不超过点云实际点数）
indices = np.random.choice(32768, 10000, replace=False)


class SamplePoints(Transform):
    """Constantly sample a fixed number of points from the point cloud."""

    def __init__(self, num_samples: int, replace=False):
        self.num_samples = num_samples
        global indices
        self.indices = indices  # 复用全局采样索引

    def apply(self, example):
        coords = np.asarray(example["coords"])
        features = np.asarray(example["features"])  # 6通道：RGB+法向量
        gt_masks = np.array(example["gt_masks"])  # [M, N]

        # 处理索引超出点云长度的情况（避免越界）
        self.indices = self.indices[self.indices < len(coords)]  # 过滤无效索引
        # 若过滤后索引不足，补充随机采样（确保数量足够）
        if len(self.indices) < self.num_samples:
            remaining = self.num_samples - len(self.indices)
            indices_2 = np.random.choice(
                len(coords), remaining, replace=False
            )
            self.indices = np.concatenate([self.indices, indices_2])

        # 对坐标和6通道特征采样（无需单独处理normals）
        example["coords"] = coords[self.indices]
        example["features"] = features[self.indices]

        # 同步采样掩码，确保与点索引匹配
        gt_masks = gt_masks[:, self.indices]
        # 处理空掩码（替换为第一个有效掩码）
        is_empty_mask = (gt_masks == 0).all(axis=1)
        if is_empty_mask.any():
            gt_masks[is_empty_mask] = gt_masks[~is_empty_mask][0]
        example["gt_masks"] = gt_masks

        return example


class SampleSingleMask(Transform):
    """ Constantly sample a single mask from the gt_masks. """

    def __init__(self, mask_id):
        self.mask_id = mask_id

    def apply(self, example):
        masks = example["gt_masks"]
        example["gt_masks"] = [masks[self.mask_id]]
        return example


class RandomSampleMask(Transform):
    """Randomly sample a fixed number of masks."""

    def __init__(self, num_samples):
        self.num_samples = num_samples

    def apply(self, example):
        masks = example["gt_masks"]
        num_masks = len(masks)
        if num_masks < self.num_samples:
            mask_idx = np.random.choice(
                num_masks, self.num_samples - num_masks, replace=False
            )
            mask_idx = np.concatenate([np.arange(num_masks), mask_idx])
        elif num_masks > self.num_samples:
            mask_idx = np.random.choice(num_masks, self.num_samples, replace=False)
        else:
            mask_idx = np.arange(num_masks)
        example["gt_masks"] = [masks[i] for i in mask_idx]
        return example


class RandomRotatePerbuate(Transform):
    def __init__(self, angle_sigma=0.06, angle_clip=0.18, normals_start_dim=3):
        self.angle_sigma = angle_sigma
        self.angle_clip = angle_clip
        self.normals_start_dim = normals_start_dim  # 法向量起始通道

    def apply(self, example):
        angles = np.clip(
            np.random.normal(0, self.angle_sigma, 3),
            -self.angle_clip,
            self.angle_clip,
        )
        rot = Rotation.from_euler("XYZ", angles)

        # 旋转坐标
        example["coords"] = rot.apply(example["coords"])

        # 旋转6通道特征中的法向量部分
        features = np.asarray(example["features"])
        normals = features[:, self.normals_start_dim:]
        features[:, self.normals_start_dim:] = rot.apply(normals)
        example["features"] = features

        return example


class RandomScale(Transform):
    def __init__(self, low, high):
        self.low = low
        self.high = high

    def apply(self, example):
        scale = np.random.uniform(self.low, self.high)
        # 只缩放坐标，不影响法向量
        example["coords"] = example["coords"] * scale
        return example


