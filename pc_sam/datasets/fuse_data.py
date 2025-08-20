from datasets import load_dataset
from torch.utils.data import Dataset
from typing import Callable
import numpy as np
from pc_sam.ply_utils import visualize_mask, visualize_pc


def rotate_point_cloud(batch_data):
    """Randomly rotate the point clouds to augument the dataset
    rotation is per shape based along up direction
    Input:
      BxNx3 array, original batch of point clouds
    Return:
      BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array(
            [[cosval, 0, sinval], [0, 1, 0], [-sinval, 0, cosval]]
        )
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def random_point_dropout(batch_pc, max_dropout_ratio=0.875):
    """batch_pc: BxNx3"""
    for b in range(batch_pc.shape[0]):
        dropout_ratio = np.random.random() * max_dropout_ratio  # 0~0.875
        drop_idx = np.where(np.random.random((batch_pc.shape[1])) <= dropout_ratio)[0]
        if len(drop_idx) > 0:
            batch_pc[b, drop_idx, :] = batch_pc[b, 0, :]  # set to the first point
    return batch_pc


def random_scale_point_cloud(batch_data, scale_low=0.8, scale_high=1):
    """Randomly scale the point cloud. Scale is per point cloud.
    Input:
        BxNx3 array, original batch of point clouds
    Return:
        BxNx3 array, scaled batch of point clouds
    """
    B, N, C = batch_data.shape
    scales = np.random.uniform(scale_low, scale_high, B)
    for batch_index in range(B):
        batch_data[batch_index, :, :] *= scales[batch_index]
    return batch_data


def shift_point_cloud(batch_data, shift_range=0.1):
    """Randomly shift point cloud. Shift is per point cloud.
    Input:
      BxNx3 array, original batch of point clouds
    Return:
      BxNx3 array, shifted batch of point clouds
    """
    B, N, C = batch_data.shape
    shifts = np.random.uniform(-shift_range, shift_range, (B, 3))
    for batch_index in range(B):
        batch_data[batch_index, :, :] += shifts[batch_index, :]
    return batch_data


def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
    """Randomly jitter points. jittering is per point.
    Input:
      BxNx3 array, original batch of point clouds
    Return:
      BxNx3 array, jittered batch of point clouds
    """
    B, N, C = batch_data.shape
    assert clip > 0
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1 * clip, clip)
    jittered_data += batch_data
    return jittered_data


def rotate_perturbation_point_cloud(batch_data, angle_sigma=0.06, angle_clip=0.18):
    """Randomly perturb the point clouds by small rotations
    Input:
      BxNx3 array, original batch of point clouds
    Return:
      BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        angles = np.clip(angle_sigma * np.random.randn(3), -angle_clip, angle_clip)
        Rx = np.array(
            [
                [1, 0, 0],
                [0, np.cos(angles[0]), -np.sin(angles[0])],
                [0, np.sin(angles[0]), np.cos(angles[0])],
            ]
        )
        Ry = np.array(
            [
                [np.cos(angles[1]), 0, np.sin(angles[1])],
                [0, 1, 0],
                [-np.sin(angles[1]), 0, np.cos(angles[1])],
            ]
        )
        Rz = np.array(
            [
                [np.cos(angles[2]), -np.sin(angles[2]), 0],
                [np.sin(angles[2]), np.cos(angles[2]), 0],
                [0, 0, 1],
            ]
        )
        R = np.dot(Rz, np.dot(Ry, Rx))
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), R)
    return rotated_data


class FuseDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        local_dir: str = None,
        transform: Callable = None,
        token: str = None,
        split="partnet+shapenet",
        mask_batch: int = 1,
        augment: bool = True,
    ):
        self.dataset = load_dataset(
            data_path,
            cache_dir=local_dir,
            token=token,
            keep_in_memory=True,
            split=split,
        )
        # self.dataset = load_dataset(
        #     "parquet",
        #     data_files="data/partnet-00000-of-00008.parquet",
        # )["train"]
        self.dataset = self.dataset.with_format("np")
        self.mask_batch = mask_batch
        self.augment = augment
        if split == "test":
            self.augment = False
            self.fix_label = True
        else:
            self.fix_label = False

    def __getitem__(self, idx):
        points = self.dataset[idx]["xyz"].astype(np.float32)
        rgb = (self.dataset[idx]["rgb"] / 255).astype(np.float32)
        labels = self.dataset[idx]["mask"]

        # normalize points
        shift = np.mean(points, axis=0)
        scale = np.max(np.linalg.norm(points - shift, ord=2, axis=1))
        points = (points - shift) / scale

        # augment
        if self.augment:
            points = random_scale_point_cloud(points[None, ...])
            points = rotate_perturbation_point_cloud(points)
            points = rotate_point_cloud(points)
            # points = jitter_point_cloud(points)
            points = points.squeeze().astype(np.float32)

        # sample mask
        if self.fix_label:
            labels = np.stack([label for label in labels if label.sum() > 0])
            labels = labels[idx % len(labels)][None, :]
        else:
            labels = [
                label
                for label in labels
                if (label.sum() > 0 and label.sum() < label.shape[0] * 0.9)
            ]
            if len(labels) == 0:
                return self.__getitem__(idx + 1 % self.__len__())
            labels = np.stack(labels)

            num_masks = labels.shape[0]
            if num_masks < self.mask_batch:
                labels = np.repeat(labels, (self.mask_batch // num_masks + 1), 0)
                num_masks = labels.shape[0]
            label_idx = np.random.choice(num_masks, [self.mask_batch])
            labels = labels[label_idx]

        data = dict(points=points, rgb=rgb, seg_labels=labels)
        return data

    def __len__(self):
        return len(self.dataset)


class FuseDatasetVal(Dataset):
    def __init__(
        self,
        data_path: str,
        local_dir: str = None,
        transform: Callable = None,
        token: str = None,
        split="partnet+shapenet",
    ):
        self.dataset = load_dataset(
            data_path,
            cache_dir=local_dir,
            token=token,
            keep_in_memory=True,
            split=split,
        )
        self.mapping_points = np.load("./mapping/points.npy")
        self.mapping_masks = np.load("./mapping/masks.npy")

    def __getitem__(self, idx):
        points = np.array(self.dataset[int(self.mapping_points[idx])]["xyz"]).astype(
            np.float32
        )
        rgb = (
            np.array(self.dataset[int(self.mapping_points[idx])]["rgb"]).astype(
                np.float32
            )
            / 255
        )
        labels = np.array(self.dataset[int(self.mapping_points[idx])]["mask"])

        # normalize points
        shift = np.mean(points, axis=0)
        scale = np.max(np.linalg.norm(points - shift, ord=2, axis=1))
        points = (points - shift) / scale

        # sample mask
        labels = labels[self.mapping_masks[idx]][None, :]
        if labels.sum() == 0:
            return self.__getitem__(0)

        data = dict(points=points, rgb=rgb, seg_labels=labels)
        return data

    def __len__(self):
        return len(self.mapping_points)



import os
import numpy as np
from torch.utils.data import Dataset


class CustomNPDDataset(Dataset):
    def __init__(self, data_root, split, transform=None, num_points=10000):
        self.data_root = data_root
        self.split = split
        self.transform = transform
        self.num_points = num_points

        # 读取split对应的txt文件（train/val/test.txt）
        split_file = os.path.join(data_root, f"{split}_scenes.txt")
        # 调试：检查split文件是否存在
        # assert os.path.exists(split_file), f"split文件不存在：{split_file}"

        # 先读取scene_names，再打印（修复顺序）
        with open(split_file, "r") as f:
            self.scene_names = [line.strip() for line in f.readlines()]  # 先定义

        # 再打印（此时self.scene_names已存在）
        # print(f"成功加载split文件：{split_file}，包含{len(self.scene_names)}个场景")

        # 检查场景文件是否存在，并打印基本信息
        # print(f"开始检查{split}集场景文件（共{len(self.scene_names)}个）：")
        for i, name in enumerate(self.scene_names):
            scene_path = os.path.join(data_root, f"{name}")
            # 检查文件是否存在（支持带或不带.npy后缀的文件）
            if not os.path.exists(scene_path):
                scene_path += ".npy"  # 尝试添加.npy后缀
            # assert os.path.exists(scene_path), f"场景文件不存在：{scene_path}"
            if i < 3:  # 只打印前3个场景的路径（避免日志过多）
                print(f"  场景{i + 1}：{scene_path}")

    def __len__(self):
        return len(self.scene_names)

    def __getitem__(self, idx):
        # 加载.npy文件（处理带或不带.npy后缀的情况）
        scene_name = self.scene_names[idx]
        scene_path = os.path.join(self.data_root, scene_name)
        if not os.path.exists(scene_path):
            scene_path += ".npy"  # 自动补充.npy后缀
        # 调试：打印当前加载的文件路径
        # print(f"\n加载第{idx}个场景：{scene_path}")

        # 加载原始数据并检查形状
        try:
            scene_data = np.load(scene_path)
        except Exception as e:
            raise RuntimeError(f"加载场景{scene_path}失败：{e}")

        # 调试：检查原始数据形状（必须是2维，且通道数为10）
        # assert scene_data.ndim == 2, f"原始数据不是2维数组，实际形状：{scene_data.shape}"
        # assert scene_data.shape[1] == 10, f"原始数据通道数错误（应为10），实际：{scene_data.shape[1]}"
        # print(f"原始数据形状：{scene_data.shape}（N={scene_data.shape[0]}个点，10通道）")

        # 解析通道并检查形状
        xyz = scene_data[:, :3].astype(np.float32)  # 坐标 (N, 3)
        rgb = scene_data[:, 3:6].astype(np.float32)  # 颜色 (N, 3)
        normal = scene_data[:, 6:9].astype(np.float32)  # 法向量 (N, 3)
        labels = scene_data[:, 9].astype(np.int32)  # 标签 (N,)

        # 调试：检查各通道形状
        # assert xyz.ndim == 2 and xyz.shape[1] == 3, f"坐标形状错误：{xyz.shape}（预期(N,3)）"
        # assert rgb.ndim == 2 and rgb.shape[1] == 3, f"RGB形状错误：{rgb.shape}（预期(N,3)）"
        # assert normal.ndim == 2 and normal.shape[1] == 3, f"法向量形状错误：{normal.shape}（预期(N,3)）"
        # assert labels.ndim == 1 and len(labels) == xyz.shape[0], f"标签形状错误：{labels.shape}（预期(N,)）"
        # print(f"解析后：坐标{xyz.shape}，RGB{rgb.shape}，法向量{normal.shape}，标签{labels.shape}")

        # 拼接6通道特征
        features = np.concatenate([rgb, normal], axis=1).astype(np.float32)
        # assert features.shape == (xyz.shape[0], 6), f"特征拼接错误：{features.shape}（预期(N,6)）"

        # 转换标签为掩码（gt_masks）
        unique_labels = np.unique(labels)
        unique_labels = unique_labels[unique_labels != -1]  # 排除无效标签
        gt_masks = np.zeros((len(unique_labels), xyz.shape[0]), dtype=bool)
        for i, label in enumerate(unique_labels):
            gt_masks[i] = (labels == label)

        # 调试：检查掩码形状
        # assert gt_masks.ndim == 2, f"掩码不是2维数组：{gt_masks.shape}"
        # assert gt_masks.shape[1] == xyz.shape[0], f"掩码点数不匹配：{gt_masks.shape[1]} vs {xyz.shape[0]}"
        # print(f"掩码形状：{gt_masks.shape}（{len(unique_labels)}个实例，{xyz.shape[0]}个点）")

        # 组织数据
        data = {
            "coords": xyz,
            "features": features,
            "gt_masks": gt_masks
        }

        # 应用变换并检查结果
        if self.transform is not None:
            # print("开始应用变换...")
            data = self.transform(data)
            # 变换后再次检查坐标形状（确保采样后仍为(N,3)）
            # assert data["coords"].ndim == 2 and data["coords"].shape[1] == 3, \
            #     f"变换后坐标形状错误：{data['coords'].shape}（预期(N,3)）"
            # print(f"变换后：坐标{data['coords'].shape}，特征{data['features'].shape}，掩码{data['gt_masks'].shape}")

        # 验证采样后点数量
        # assert len(data["coords"]) == self.num_points, \
        #     f"采样后点数量错误：预期{self.num_points}，实际{len(data['coords'])}"
        # print(f"数据加载完成（第{idx}个场景）")
        return data