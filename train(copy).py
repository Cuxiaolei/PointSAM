import argparse
import os
import json
from datetime import datetime
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path

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


def build_datasets(cfg):
    if cfg.dataset.name == "CustomNPY":
        print("transforms 配置列表：")
        for i, t in enumerate(cfg.transforms):
            print(f"第 {i} 个变换的配置类型：{type(t)}，内容：{t}")
        # 解析transforms
        transforms = None
        # 正确（transforms 与 dataset 同级）
        if hasattr(cfg, "transforms") and cfg.transforms is not None:
            # 直接使用已实例化的变换列表，组合成Compose
            transform_list = cfg.transforms  # 关键：不再调用instantiate
            transforms = Compose(transform_list)
            print(f"已使用已实例化的变换链：{[t.__class__.__name__ for t in transform_list]}")

        random_sample_transform = cfg.transforms[2]
        num_samples = random_sample_transform.num_samples
        return CustomNPDDataset(
            data_root=cfg.dataset.path,
            split=cfg.dataset.split,
            transform=transforms,
            # 从全局配置获取num_samples（关键修改）
            num_points=num_samples
        )
    else:
        return build_dataset(cfg)

# NOTE: We separately instantiate each component for fine-grained control.
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="large", help="path to config file"
    )
    parser.add_argument("--config_dir", type=str, default="configs")
    args, unknown_args = parser.parse_known_args()

    # ---------------------------------------------------------------------------- #
    # Load configuration
    # ---------------------------------------------------------------------------- #
    with hydra.initialize(args.config_dir, version_base=None):
        cfg = hydra.compose(config_name=args.config, overrides=unknown_args)
        OmegaConf.resolve(cfg)
        # print(OmegaConf.to_yaml(cfg))

    # Prepare (flat) hyperparameters for logging
    hparams = {
        "lr": cfg.lr,
        "weight_decay": cfg.weight_decay,
        "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
        "batch_size": cfg.train_dataloader.batch_size * cfg.gradient_accumulation_steps,
    }

    # Check cuda and cudnn settings
    torch.backends.cudnn.benchmark = True
    print("flash_sdp_enabled:", torch.backends.cuda.flash_sdp_enabled())
    print("mem_efficient_sdp_enabled:", torch.backends.cuda.mem_efficient_sdp_enabled())
    print("math_sdp_enabled:", torch.backends.cuda.math_sdp_enabled())

    seed = cfg.get("seed", 42)

    # ---------------------------------------------------------------------------- #
    # Setup model
    # ---------------------------------------------------------------------------- #
    set_seed(seed)
    model: PointCloudSAM = hydra.utils.instantiate(cfg.model)
    model.apply(replace_with_fused_layernorm)

    # ---------------------------------------------------------------------------- #
    # Initialize with pre-trained weights if provided
    # ---------------------------------------------------------------------------- #
    if cfg.pretrained_ckpt_path:
        print("Loading pretrained weight from", cfg.pretrained_ckpt_path)
        pretrained = torch.load(cfg.pretrained_ckpt_path)
        # Hardcoded for Uni3D
        state_dict = {}
        for name in pretrained["module"].keys():
            if "point_encoder.encoder2trans" in name:
                # print(name)
                suffix = name[len("point_encoder.encoder2trans.") :]
                state_dict[f"patch_proj.{suffix}"] = pretrained["module"][name]
                # print(name, pretrained["module"][name].shape)
            if "point_encoder.pos_embed" in name:
                # print(name)
                suffix = name[len("point_encoder.pos_embed.") :]
                state_dict[f"pos_embed.{suffix}"] = pretrained["module"][name]
            if "point_encoder.visual" in name:
                # print(name)
                suffix = name[len("point_encoder.visual.") :]
                state_dict[f"transformer.{suffix}"] = pretrained["module"][name]
        missing_keys = model.pc_encoder.load_state_dict(state_dict, strict=False)
        print(missing_keys)

    # ---------------------------------------------------------------------------- #
    # Setup dataloaders
    # ---------------------------------------------------------------------------- #
    train_dataset_cfg = hydra.utils.instantiate(cfg.train_dataset)
    train_dataset = build_datasets(train_dataset_cfg)

    train_dataloader = DataLoader(
        train_dataset,
        **cfg.train_dataloader,
        worker_init_fn=worker_init_fn,
        generator=torch.Generator().manual_seed(seed),
    )

    if cfg.val_freq > 0:
        val_dataset_cfg = hydra.utils.instantiate(cfg.val_dataset)
        val_dataset = build_datasets(val_dataset_cfg)
        val_dataloader = DataLoader(
            val_dataset, **cfg.val_dataloader, worker_init_fn=worker_init_fn
        )

    # ---------------------------------------------------------------------------- #
    # Setup optimizer
    # ---------------------------------------------------------------------------- #
    params = []
    for name, module in model.named_children():
        # NOTE: Different learning rates can be set for different modules
        if name == "pc_encoder":
            params += [{"params": module.parameters(), "lr": cfg.lr}]
        else:
            params += [{"params": module.parameters(), "lr": cfg.lr}]

    optimizer = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = hydra.utils.instantiate(cfg.scheduler, optimizer=optimizer)

    # criterion = Criterion()
    criterion = hydra.utils.instantiate(cfg.loss)

    # ---------------------------------------------------------------------------- #
    # Initialize accelerator
    # ---------------------------------------------------------------------------- #
    project_config = ProjectConfiguration(
        cfg.project_dir, automatic_checkpoint_naming=True, total_limit=1
    )
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        project_config=project_config,
        kwargs_handlers=[ddp_kwargs],
        log_with=[],
    )
    model, optimizer, train_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, scheduler
    )
    if cfg.val_freq > 0:
        val_dataloader = accelerator.prepare(val_dataloader)

    accelerator.print(OmegaConf.to_yaml(cfg))

    if cfg.log_with == "":
        accelerator.init_trackers(
            project_name=cfg.get("project_name", "pointcloud-sam"),
            config=hparams,
            init_kwargs={"log": {"name": cfg.run_name}},
        )

    # 初始化本地日志
    log_dir = os.path.join(cfg.project_dir, "metrics")
    os.makedirs(log_dir, exist_ok=True)
    train_log_path = os.path.join(log_dir, "train_metrics.jsonl")
    val_log_path = os.path.join(log_dir, "val_metrics.jsonl")
    print(f"日志文件将保存至: {log_dir}")

    # 移除wandb相关初始化代码
    if cfg.log_with == "":
        accelerator.print(f"已禁用wandb，使用本地日志记录")
    else:
        print(f"没有禁用wandb，cfg.log_with为{cfg.log_with}")
    # Define validation function


    @torch.no_grad()
    def validate(current_epoch, current_step):
        model.eval()
        epoch_ious = defaultdict(list)
        pbar = tqdm(total=len(val_dataloader), desc=f"Validation Epoch {current_epoch + 1}")

        for data in val_dataloader:
            outputs = model(**data, is_eval=True)
            gt_masks = data["gt_masks"].flatten(0, 1)

            # Update metrics
            for i_iter in range(len(outputs)):
                if i_iter == 0:
                    all_masks = outputs[0]["masks"]  # [B*M, C, N]
                    all_ious = compute_iou(
                        all_masks, gt_masks.unsqueeze(1).expand_as(all_masks)
                    )
                    best_iou = all_ious.max(dim=1).values
                    epoch_ious["best"].extend(best_iou.tolist())
                iou = compute_iou(outputs[i_iter]["prompt_masks"], gt_masks)
                epoch_ious[i_iter].extend(iou.tolist())

            metrics = {
                f"iou({i_iter})": np.mean(iou) for i_iter, iou in epoch_ious.items()
            }
            sub_metrics = {
                f"iou({i_iter})": metrics[f"iou({i_iter})"]
                for i_iter in [0, len(outputs) - 1]
            }
            pbar.set_postfix(sub_metrics)
            pbar.update(1)

        pbar.close()

        # 打印验证指标
        print(f"\n===== 验证集指标 (Epoch: {current_epoch + 1}, Step: {current_step}) =====")
        for key, value in metrics.items():
            print(f"  {key}: {value:.6f}")

        # 保存验证日志
        val_log_entry = {
            "epoch": current_epoch + 1,
            "global_step": current_step,
            "timestamp": datetime.now().isoformat(), **metrics
        }
        with open(val_log_path, "a") as f:
            f.write(json.dumps(val_log_entry) + "\n")

        return metrics

        # ---------------------------------------------------------------------------- #
        # Training loop
        # ---------------------------------------------------------------------------- #
        step = 0  # Number of batch steps
        global_step = 0  # Number of optimization steps
        start_epoch = 0

        # Restore state
        ckpt_dir = Path(accelerator.project_dir, "checkpoints")
        if ckpt_dir.exists():
            accelerator.load_state()
            global_step = scheduler.scheduler.last_epoch // accelerator.state.num_processes
            get_epoch_fn = lambda x: int(x.name.split("_")[-1])
            last_ckpt_dir = sorted(ckpt_dir.glob("checkpoint_*"), key=get_epoch_fn)[-1]
            start_epoch = get_epoch_fn(last_ckpt_dir) + 1
            accelerator.project_configuration.iteration = start_epoch

        for epoch in range(start_epoch, cfg.max_epochs):
            model.train()
            print(f"\n===== 开始训练 Epoch {epoch + 1}/{cfg.max_epochs} =====")
            pbar = tqdm(total=len(train_dataloader), desc=f"Training Epoch {epoch + 1}")

            for data in train_dataloader:
                flag = (step + 1) % cfg.gradient_accumulation_steps == 0

                ctx = nullcontext if flag else accelerator.no_sync
                with ctx(model):
                    outputs = model(**data)
                    gt_masks = data["gt_masks"].flatten(0, 1)  # [B*M, N]
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

                    # Compute metrics
                    with torch.no_grad():
                        metrics = dict(loss=loss.item())
                        for i_iter in [0, len(outputs) - 1]:
                            pred_masks = aux[i_iter]["best_masks"] > 0
                            is_correct = pred_masks == gt_masks
                            acc = is_correct.float().mean()
                            fg_acc = is_correct[gt_masks == 1].float().mean()
                            bg_acc = is_correct[gt_masks == 0].float().mean()
                            metrics[f"acc({i_iter})"] = acc.item()
                            metrics[f"fg_acc({i_iter})"] = fg_acc.item()
                            metrics[f"bg_acc({i_iter})"] = bg_acc.item()

                            iou = aux[i_iter]["iou"].mean()
                            metrics[f"iou({i_iter})"] = iou.item()

                            # Loss breakdown
                            for k, v in aux[i_iter].items():
                                if k.startswith("loss"):
                                    metrics[f"{k}({i_iter})"] = v.item()

                    # 打印训练指标
                    if global_step % 10 == 0:  # 每10步打印一次详细指标
                        print(f"\n训练步骤 {global_step} 指标:")
                        for key, value in metrics.items():
                            print(f"  {key}: {value:.6f}")

                    # 保存训练日志
                    train_log_entry = {
                        "epoch": epoch + 1,
                        "global_step": global_step,
                        "timestamp": datetime.now().isoformat(), **metrics
                    }
                    with open(train_log_path, "a") as f:
                        f.write(json.dumps(train_log_entry) + "\n")

                    # 更新进度条
                    sub_metrics = {
                        k: v
                        for k, v in metrics.items()
                        if k.startswith("acc") or k.startswith("iou") or k == "loss"
                    }
                    pbar.set_postfix(sub_metrics)

                    global_step += 1

                pbar.update(1)
                step += 1
                if global_step >= cfg.max_steps:
                    break

            pbar.close()
            print(f"===== 完成训练 Epoch {epoch + 1} =====")

            # Save state
            if (epoch + 1) % cfg.get("save_freq", 1) == 0:
                accelerator.save_state()
                print(f"已保存模型 checkpoint (Epoch {epoch + 1})")

            # 验证步骤
            if cfg.val_freq > 0 and (epoch + 1) % cfg.val_freq == 0:
                torch.cuda.empty_cache()
                with accelerator.no_sync(model):
                    metrics = validate(epoch, global_step)
                torch.cuda.empty_cache()

            if global_step >= cfg.max_steps:
                break

        print("训练完成!")
        accelerator.end_training()

@torch.no_grad()
def get_wandb_object_3d(xyz, rgb, gt_masks, pred_masks, prompt_coords, prompt_labels):
    pcds = []
    xyz = xyz[0].cpu().numpy()  # [N, 3]
    rgb = (rgb[0].cpu().numpy() * 0.5 + 0.5) * 255  # [N, 3]
    gt_mask = gt_masks[0].cpu().numpy()  # [N]

    input_pcd = np.concatenate([xyz, rgb], axis=1)
    pcds.append(wandb.Object3D(input_pcd))

    gt_pcd = np.concatenate([xyz, gt_mask[:, None]], axis=1)
    pcds.append(wandb.Object3D(gt_pcd))

    # Only visualize the first sample
    for i, pred_mask in enumerate(pred_masks):
        pred_mask = pred_mask[0].cpu().numpy()
        # pred_pcd = np.concatenate([xyz, pred_mask[:, None]], axis=1)
        xyz2 = np.concatenate([xyz, prompt_coords[i][0].cpu().numpy()])
        pred_mask = np.concatenate([pred_mask, prompt_labels[i][0].cpu().numpy() + 2])
        pred_pcd = np.concatenate([xyz2, pred_mask[:, None]], axis=1)
        pcds.append(wandb.Object3D(pred_pcd))

    return pcds


if __name__ == "__main__":
    main()
