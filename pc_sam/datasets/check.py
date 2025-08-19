# 单独运行的检查脚本
import numpy as np
from transforms import Compose, NormalizePoints, NormalizeColor, RandomSample, ToTensor

# 加载异常场景数据
scene_path = "/root/autodl-tmp/data/data_s3dis_pointNeXt/merged/Area_5.npy"
scene_data = np.load(scene_path)
print(f"Area_5原始形状：{scene_data.shape}")  # 应是(N,10)

# 解析坐标
xyz = scene_data[:, :3].astype(np.float32)
print(f"解析后坐标形状：{xyz.shape}")  # 应是(N,3)

# 应用变换链
transforms = Compose([
    NormalizePoints(),
    NormalizeColor(mean=0.5, std=0.5),
    RandomSample(num_samples=10000),
    ToTensor()
])
data = {"coords": xyz}  # 构造完整数据
try:
    data = transforms(data)
    print(f"变换后坐标形状：{data['coords'].shape}")  # 预期(10000,3)
except Exception as e:
    print(f"变换中出错：{e}")