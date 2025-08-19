import numpy as np
from transforms import (
    Compose, NormalizePoints, NormalizeColor, RandomSample, ToTensor
)

# 加载数据
scene_path = "/root/autodl-tmp/data/data_s3dis_pointNeXt/merged/Area_5.npy"
scene_data = np.load(scene_path)
print(f"Area_5原始形状：{scene_data.shape}")

# 解析数据（与数据集逻辑一致）
xyz = scene_data[:, :3].astype(np.float32)  # (N, 3)
rgb = scene_data[:, 3:6].astype(np.float32)
normal = scene_data[:, 6:9].astype(np.float32)
labels = scene_data[:, 9].astype(np.int32)
features = np.concatenate([rgb, normal], axis=1)  # (N, 6)
unique_labels = np.unique(labels)
unique_labels = unique_labels[unique_labels != -1]
gt_masks = np.zeros((len(unique_labels), xyz.shape[0]), dtype=bool)
for i, label in enumerate(unique_labels):
    gt_masks[i] = (labels == label)

data = {"coords": xyz, "features": features, "gt_masks": gt_masks}
print(f"初始坐标形状：{data['coords'].shape}")  # 预期 (294362, 3)

# 逐个应用变换并检查形状
transforms = [
    ("NormalizePoints", NormalizePoints()),
    ("NormalizeColor", NormalizeColor(mean=0.5, std=0.5)),
    ("RandomSample", RandomSample(num_samples=10000)),
    ("ToTensor", ToTensor())
]

current_data = data.copy()
for name, transform in transforms:
    try:
        current_data = transform(current_data)  # 单独应用当前变换
        coords_shape = current_data["coords"].shape
        assert current_data["coords"].ndim == 2 and coords_shape[1] == 3, \
            f"变换 {name} 后坐标形状错误：{coords_shape}"
        print(f"变换 {name} 后：坐标形状 {coords_shape}")
    except Exception as e:
        print(f"变换 {name} 出错：{e}")
        break