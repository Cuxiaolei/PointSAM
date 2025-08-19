import numpy as np
from transforms import Compose, NormalizePoints, NormalizeColor, RandomSample, ToTensor

# 加载异常场景数据
scene_path = "/root/autodl-tmp/data/data_s3dis_pointNeXt/merged/Area_5.npy"
scene_data = np.load(scene_path)
print(f"Area_5原始形状：{scene_data.shape}")  # 应是(N,10)

# 解析坐标、特征、掩码（与CustomNPDDataset逻辑一致）
xyz = scene_data[:, :3].astype(np.float32)  # 坐标 (N,3)
rgb = scene_data[:, 3:6].astype(np.float32)  # RGB (N,3)
normal = scene_data[:, 6:9].astype(np.float32)  # 法向量 (N,3)
labels = scene_data[:, 9].astype(np.int32)  # 标签 (N,)

# 正确构造6通道特征
features = np.concatenate([rgb, normal], axis=1).astype(np.float32)  # (N,6)

# 正确构造掩码（gt_masks: [M, N]，M为实例数）
unique_labels = np.unique(labels)
unique_labels = unique_labels[unique_labels != -1]  # 排除无效标签
gt_masks = np.zeros((len(unique_labels), xyz.shape[0]), dtype=bool)
for i, label in enumerate(unique_labels):
    gt_masks[i] = (labels == label)

# 构造完整数据字典（无占位符）
data = {
    "coords": xyz,
    "features": features,  # 正确的(N,6)数组
    "gt_masks": gt_masks   # 正确的(M,N)布尔数组
}

# 应用变换链
transforms = Compose([
    NormalizePoints(),
    NormalizeColor(mean=0.5, std=0.5),
    RandomSample(num_samples=10000),
    ToTensor()
])

try:
    transformed_data = transforms(data)
    print(f"变换后坐标形状：{transformed_data['coords'].shape}")  # 预期(10000,3)
    print("变换成功，无形状异常")
except Exception as e:
    print(f"变换中出错：{e}")