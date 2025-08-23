import argparse
import os
import json
from datetime import datetime
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
import csv
from typing import Dict, Any

import hydra
import numpy as np
import torch
import torch.nn as nn
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.accelerator import ProjectConfiguration
from accelerate.utils import set_seed, tqdm
from datasets import DatasetDict, load_dataset, load_from_disk
from omegaconf import OmegaConf
from torch.utils.data import ConcatDataset, DataLoader

from pc_sam.datasets.transforms import Compose
from pc_sam.model.loss import compute_iou
from pc_sam.model.pc_sam import PointCloudSAM
from pc_sam.utils.torch_utils import replace_with_fused_layernorm, worker_init_fn
from hydra.utils import instantiate
from torchvision.transforms import Compose
from pc_sam.datasets.fuse_data import CustomNPDDataset  # 根据实际路径调整导入
import omegaconf
from omegaconf.listconfig import ListConfig

def build_dataset(cfg):
    if os.path.exists(cfg.dataset.path):
        keep_in_memory = cfg.get("keep_in_memory", False)
        dataset = load_from_disk(cfg.dataset.path, keep_in_memory=keep_in_memory)
        split = cfg.dataset.get("split", "train")
        dataset = dataset[split]
    else:
        dataset = load_dataset(**cfg.dataset)

    dataset = dataset.rename_columns(
        {"xyz": "coords", "rgb": "features", "mask": "gt_masks"}
    )
    dataset = dataset.select_columns(["coords", "features", "gt_masks"])

    dataset.set_transform(Compose(cfg.transforms))

    if "repeats" in cfg:
        from torch.utils.data import Subset  # fmt: skip
        dataset = Subset(dataset, list(range(len(dataset))) * cfg.repeats)

    return dataset


def classwise_metrics(pred_masks: torch.Tensor, gt_labels: torch.Tensor, num_classes: int = 3):
    """
    计算每个类别的IoU、准确率、整体准确率（OA）和平均IoU（mIoU）
    Args:
        pred_masks: 预测的类别掩码，形状 [N]（每个点的预测类别，0/1/2）
        gt_labels: 真实类别标签，形状 [N]（每个点的真实类别，0/1/2）
        num_classes: 类别数（根据实际数据集调整）
    Returns:
        字典包含：
        - iou: 每个类别的IoU，形状 [num_classes]
        - acc: 每个类别的准确率，形状 [num_classes]
        - oa: 整体准确率
        - miou: 平均IoU（所有类别IoU的均值）
    """
    iou = torch.zeros(num_classes, device=pred_masks.device)
    acc = torch.zeros(num_classes, device=pred_masks.device)
    total_correct = 0
    total = 0

    for c in range(num_classes):
        # 计算当前类别的TP、FP、FN
        pred_c = (pred_masks == c)
        gt_c = (gt_labels == c)

        tp = (pred_c & gt_c).sum().float()
        fp = (pred_c & ~gt_c).sum().float()
        fn = (~pred_c & gt_c).sum().float()
        total_c = gt_c.sum().float()

        # 计算IoU（避免除以0）
        iou_denominator = tp + fp + fn
        iou[c] = tp / iou_denominator if iou_denominator > 0 else 0.0

        # 计算类别准确率
        acc[c] = tp / total_c if total_c > 0 else 0.0

        # 累计整体指标
        total_correct += tp
        total += total_c

    # 计算整体准确率（OA）和平均IoU（mIoU）
    oa = total_correct / total if total > 0 else 0.0
    miou = iou.mean().item()  # 所有类别IoU的平均值

    return {
        "iou": iou,
        "acc": acc,
        "oa": oa,
        "miou": miou
    }

def masks_to_labels(gt_masks: torch.Tensor) -> torch.Tensor:
    """将掩码转换为类别标签（每个点属于唯一类别）"""
    # 输入gt_masks形状为 [M, N]，输出形状为 [N]（0~M-1）
    labels = torch.argmax(gt_masks.float(), dim=0)
    return labels



def build_datasets(cfg):
    if cfg.dataset.name == "CustomNPY":
        print("transforms 配置列表：")
        for i, t in enumerate(cfg.transforms):
            print(f"第 {i} 个变换的配置类型：{type(t)}，内容：{t}")
        # 解析transforms
        transforms = None
        if hasattr(cfg, "transforms") and cfg.transforms is not None:
            transform_list = cfg.transforms
            transforms = Compose(transform_list)
            print(f"已使用已实例化的变换链：{[t.__class__.__name__ for t in transform_list]}")

        random_sample_transform = cfg.transforms[2]
        num_samples = random_sample_transform.num_samples
        return CustomNPDDataset(
            data_root=cfg.dataset.path,
            split=cfg.dataset.split,
            transform=transforms,
            num_points=num_samples
        )
    else:
        return build_dataset(cfg)


def save_config(cfg, log_dir):
    """保存配置文件到本地日志目录"""
    config_path = os.path.join(log_dir, "full_config.yaml")
    with open(config_path, "w") as f:
        f.write(OmegaConf.to_yaml(cfg))
    print(f"配置文件已保存至: {config_path}")


def init_log_files(log_dir):
    os.makedirs(log_dir, exist_ok=True)

    train_log_path = os.path.join(log_dir, "train_metrics.csv")
    val_log_path = os.path.join(log_dir, "val_metrics.csv")

    # 训练日志表头（包含类别IoU、mIoU等）
    if not os.path.exists(train_log_path):
        with open(train_log_path, "w", newline="") as f:
            writer = csv.writer(f)
            headers = ["step", "loss"]
            # 为每个输出分支添加指标（假设3个类别）
            for i_iter in [0, -1]:  # 关注第0个和最后一个分支
                for c in range(3):
                    headers.append(f"iou_{c}({i_iter})")  # 类别c的IoU
                    headers.append(f"acc_{c}({i_iter})")  # 类别c的准确率
                headers.append(f"oa({i_iter})")          # 整体准确率
                headers.append(f"miou({i_iter})")        # 平均IoU
            writer.writerow(headers)

    # 验证日志表头
    if not os.path.exists(val_log_path):
        with open(val_log_path, "w", newline="") as f:
            writer = csv.writer(f)
            headers = ["step"]
            for i_iter in [0, -1]:
                for c in range(3):
                    headers.append(f"iou_{c}({i_iter})")
                    headers.append(f"acc_{c}({i_iter})")
                headers.append(f"oa({i_iter})")
                headers.append(f"miou({i_iter})")
            headers.append("miou(best)")  # 最佳分支的mIoU
        writer.writerow(headers)

    return train_log_path, val_log_path

def log_to_csv(log_path: str, step: int, metrics: Dict[str, Any]):
    """将指标写入CSV文件"""
    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        # 提取表头顺序
        with open(log_path, "r") as rf:
            reader = csv.reader(rf)
            headers = next(reader)

        # 按照表头顺序准备数据
        row = [step]
        for header in headers[1:]:  # 跳过step列
            row.append(metrics.get(header, ""))

        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="large", help="path to config file"
    )
    parser.add_argument("--config_dir", type=str, default="configs")
    args, unknown_args = parser.parse_known_args()

    # 加载配置
    with hydra.initialize(args.config_dir, version_base=None):
        cfg = hydra.compose(config_name=args.config, overrides=unknown_args)
        OmegaConf.resolve(cfg)

        # 强制禁用wandb，使用本地日志
        cfg.log_with = None
        print("已禁用wandb，使用本地日志记录")

    # 准备超参数
    hparams = {
        "lr": cfg.lr,
        "weight_decay": cfg.weight_decay,
        "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
        "batch_size": cfg.train_dataloader.batch_size * cfg.gradient_accumulation_steps,
    }

    # 检查cuda和cudnn设置
    torch.backends.cudnn.benchmark = True
    print("flash_sdp_enabled:", torch.backends.cuda.flash_sdp_enabled())
    print("mem_efficient_sdp_enabled:", torch.backends.cuda.mem_efficient_sdp_enabled())
    print("math_sdp_enabled:", torch.backends.cuda.math_sdp_enabled())

    seed = cfg.get("seed", 42)
    set_seed(seed)

    # 初始化模型
    model: PointCloudSAM = hydra.utils.instantiate(cfg.model)
    model.apply(replace_with_fused_layernorm)

    # 加载预训练权重
    if cfg.pretrained_ckpt_path:
        print("Loading pretrained weight from", cfg.pretrained_ckpt_path)
        pretrained = torch.load(cfg.pretrained_ckpt_path)
        state_dict = {}
        for name in pretrained["module"].keys():
            if "point_encoder.encoder2trans" in name:
                suffix = name[len("point_encoder.encoder2trans."):]
                state_dict[f"patch_proj.{suffix}"] = pretrained["module"][name]
            if "point_encoder.pos_embed" in name:
                suffix = name[len("point_encoder.pos_embed."):]
                state_dict[f"pos_embed.{suffix}"] = pretrained["module"][name]
            if "point_encoder.visual" in name:
                suffix = name[len("point_encoder.visual."):]
                state_dict[f"transformer.{suffix}"] = pretrained["module"][name]
        missing_keys = model.pc_encoder.load_state_dict(state_dict, strict=False)
        print(missing_keys)

    # 准备数据集
    train_dataset_cfg = hydra.utils.instantiate(cfg.train_dataset)
    train_dataset = build_datasets(train_dataset_cfg)

    train_dataloader = DataLoader(
        train_dataset,
        **cfg.train_dataloader,
        worker_init_fn=worker_init_fn,
        generator=torch.Generator().manual_seed(seed),
    )

    val_dataloader = None
    if cfg.val_freq > 0:
        val_dataset_cfg = hydra.utils.instantiate(cfg.val_dataset)
        val_dataset = build_datasets(val_dataset_cfg)
        val_dataloader = DataLoader(
            val_dataset, **cfg.val_dataloader, worker_init_fn=worker_init_fn
        )

    # 优化器和调度器
    params = []
    for name, module in model.named_children():
        if name == "pc_encoder":
            params += [{"params": module.parameters(), "lr": cfg.lr}]
        else:
            params += [{"params": module.parameters(), "lr": cfg.lr}]

    optimizer = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = hydra.utils.instantiate(cfg.scheduler, optimizer=optimizer)
    criterion = hydra.utils.instantiate(cfg.loss)

    # 初始化加速器
    project_config = ProjectConfiguration(
        cfg.project_dir, automatic_checkpoint_naming=True, total_limit=1
    )
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        project_config=project_config,
        kwargs_handlers=[ddp_kwargs],
        log_with=cfg.log_with,  # 已设置为None，禁用wandb
    )
    model, optimizer, train_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, scheduler
    )
    if cfg.val_freq > 0 and val_dataloader is not None:
        val_dataloader = accelerator.prepare(val_dataloader)

    accelerator.print(OmegaConf.to_yaml(cfg))

    # 初始化本地日志
    log_dir = cfg.log_dir
    os.makedirs(log_dir, exist_ok=True)
    save_config(cfg, log_dir)
    train_log_path, val_log_path = init_log_files(log_dir)

    # 验证函数
    @torch.no_grad()
    def validate():
        model.eval()
        epoch_metrics = defaultdict(list)  # 存储所有样本的指标
        pbar = tqdm(total=len(val_dataloader), desc="Validation")

        for data in val_dataloader:
            outputs = model(**data, is_eval=True)
            gt_masks = data["gt_masks"].flatten(0, 1)  # 假设形状 [3, N]
            gt_labels = masks_to_labels(gt_masks)  # 转换为 [N] 的 0/1/2

            for i_iter in range(len(outputs)):
                # 预测标签（根据模型输出调整）
                pred_masks = outputs[i_iter]["prompt_masks"]  # 假设 [3, N]
                pred_labels = torch.argmax(pred_masks, dim=0)  # [N]

                # 计算类别指标
                class_metrics = classwise_metrics(pred_labels, gt_labels, num_classes=3)

                # 记录每个类别的指标
                for c in range(3):
                    epoch_metrics[f"iou_{c}({i_iter})"].append(class_metrics["iou"][c].item())
                    epoch_metrics[f"acc_{c}({i_iter})"].append(class_metrics["acc"][c].item())
                epoch_metrics[f"oa({i_iter})"].append(class_metrics["oa"].item())

            # 实时更新进度条
            sub_metrics = {f"iou_0({i_iter})": np.mean(epoch_metrics[f"iou_0({i_iter})"])}  # 示例
            pbar.set_postfix(sub_metrics)
            pbar.update(1)

        pbar.close()

        # 计算平均指标
        avg_metrics = {}
        for key, values in epoch_metrics.items():
            avg_metrics[key] = np.mean(values)

        print("\n验证结果:")
        for key, value in avg_metrics.items():
            print(f"  {key}: {value:.4f}")

        return avg_metrics

    # 训练循环
    step = 0  # 批次步骤数
    global_step = 0  # 优化步骤数
    start_epoch = 0

    # 恢复状态
    ckpt_dir = Path(accelerator.project_dir, "checkpoints")
    if ckpt_dir.exists():
        # 允许加载OmegaConf的ListConfig类型
        with torch.serialization.safe_globals([ListConfig]):
            try:
                accelerator.load_state()
                global_step = scheduler.scheduler.last_epoch // accelerator.state.num_processes
                get_epoch_fn = lambda x: int(x.name.split("_")[-1])
                last_ckpt_dir = sorted(ckpt_dir.glob("checkpoint_*"), key=get_epoch_fn)[-1]
                start_epoch = get_epoch_fn(last_ckpt_dir) + 1
                accelerator.project_configuration.iteration = start_epoch
            except Exception as e:
                print(f"检查点加载失败，将从头开始训练: {str(e)}")
    for epoch in range(start_epoch, cfg.max_epochs):
        model.train()
        pbar = tqdm(total=len(train_dataloader), desc=f"Epoch {epoch + 1}/{cfg.max_epochs}")
        epoch_train_metrics = defaultdict(list)  # 收集整个epoch的指标

        for data in train_dataloader:
            flag = (step + 1) % cfg.gradient_accumulation_steps == 0

            ctx = nullcontext if flag else accelerator.no_sync
            with ctx(model):
                outputs = model(**data)
                gt_masks = data["gt_masks"].flatten(0, 1)
                loss, aux = criterion(outputs, gt_masks)
                accelerator.backward(loss / cfg.gradient_accumulation_steps)

            if flag:
                if cfg.max_grad_value:
                    nn.utils.clip_grad.clip_grad_value_(
                        model.parameters(), cfg.max_grad_value
                    )
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

                # 计算指标
                # 训练循环中计算指标的部分（替换原有代码）
                with torch.no_grad():
                    metrics = dict(loss=loss.item())
                    # 转换真实掩码为类别标签（[M, N] -> [N]）
                    gt_labels = masks_to_labels(gt_masks)  # gt_masks是[M, N]的布尔张量

                    for i_iter in [0, len(outputs) - 1]:
                        # 预测掩码转换为类别标签（[M, N] -> [N]）
                        pred_masks = aux[i_iter]["best_masks"]  # 假设形状为[M, N]
                        pred_labels = torch.argmax(pred_masks, dim=0)  # [N]的类别标签

                        # 计算类别指标（包含mIoU）
                        class_metrics = classwise_metrics(pred_labels, gt_labels, num_classes=3)

                        # 记录每个类别的IoU和准确率
                        for c in range(3):
                            metrics[f"iou_{c}({i_iter})"] = class_metrics["iou"][c].item()
                            metrics[f"acc_{c}({i_iter})"] = class_metrics["acc"][c].item()
                        # 记录OA和mIoU
                        metrics[f"oa({i_iter})"] = class_metrics["oa"].item()
                        metrics[f"miou({i_iter})"] = class_metrics["miou"]

                        # 保留原有损失分解（如果需要）
                        for k, v in aux[i_iter].items():
                            if k.startswith("loss"):
                                metrics[f"{k}({i_iter})"] = v.item()

                # 记录指标到列表，用于epoch平均
                for k, v in metrics.items():
                    epoch_train_metrics[k].append(v)

                # 实时打印指标
                sub_metrics = {
                    k: v
                    for k, v in metrics.items()
                    if k.startswith("acc") or k.startswith("iou") or k == "loss"
                }
                pbar.set_postfix(sub_metrics)

                # 保存到CSV
                log_to_csv(train_log_path, global_step, metrics)

                global_step += 1

            pbar.update(1)
            step += 1
            if global_step >= cfg.max_steps:
                break

        pbar.close()

        # 打印epoch平均指标
        print(f"\nEpoch {epoch + 1} 训练平均指标:")
        for k in ["loss", "acc(0)", "iou(0)", "acc(-1)", "iou(-1)"]:
            if k in epoch_train_metrics:
                print(f"  {k}: {np.mean(epoch_train_metrics[k]):.4f}")

        # 保存状态
        if (epoch + 1) % cfg.get("save_freq", 1) == 0:
            accelerator.save_state()
            print(f"已保存 checkpoint 到: {ckpt_dir}")

        # 验证
        if cfg.val_freq > 0 and (epoch + 1) % cfg.val_freq == 0 and val_dataloader is not None:
            torch.cuda.empty_cache()
            with accelerator.no_sync(model):
                val_metrics = validate()
            torch.cuda.empty_cache()

            # 保存验证指标
            log_to_csv(val_log_path, global_step, val_metrics)

        if global_step >= cfg.max_steps:
            break

    accelerator.end_training()
    print(f"训练完成，日志文件保存至: {log_dir}")


if __name__ == "__main__":
    main()