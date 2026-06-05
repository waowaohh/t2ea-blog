---
title: "核心类、函数与优化变量字典"
date: 2025-06-05
draft: false
math: true
description: "T2EA 项目所有网络模块、损失函数、数据集类、训练脚本关键变量与工具函数的完整参考手册"
tags: ["API参考", "网络模块", "损失函数", "数据集", "工具函数"]
categories: ["参考手册"]
weight: 6
---

## 6.1 网络模块

### `Taylor_Encoder`（`network/TEM.py`）

| 属性/方法 | 类型 | 说明 |
|-----------|------|------|
| `self.base` | nn.Sequential | 零阶编码器：Conv2d(1,32,5)→LeakyReLU→ResB(32,64)→ResB(64,32)→ResB(32,1) |
| `self.gradient` | nn.Sequential | 高阶编码器：Conv2d(2,8,5)→LeakyReLU→RDBLOCK(8→16)→RDBLOCK(16→32)→RDBLOCK(32→64)→Conv2d(64,1,5)→LeakyReLU |
| `forward(input, n)` | 方法 | 输入 `[B,1,H,W]`，返回 `(result, y_list)`。`result` 是泰勒重建，`y_list` 是 n+1 个分量 |

### `FusionModel`（`network/FusionNet.py`）

| 属性/方法 | 类型 | 说明 |
|-----------|------|------|
| `self.Net` | FusionNetwork | 单层融合网络，处理一对 ir[i]/vis[i] |
| `forward(ir, vis)` | 方法 | `ir`/`vis` 是长度为 n+1 的 tensor 列表，返回逆泰勒加权融合结果 |

### `FusionNetwork`（`network/FusionNet.py`）

| 属性 | Shape 变换 | 说明 |
|------|-----------|------|
| `vis_conv` | [B,1,H,W]→[B,16,H,W] | 可见光初始卷积 |
| `vis_rgbd1` | [B,16,H,W]→[B,32,H,W] | RGBD 模块 1（Dense+Sobel） |
| `vis_rgbd2` | [B,32,H,W]→[B,48,H,W] | RGBD 模块 2 |
| `inf_conv/rgbd1/rgbd2` | 同上 | 红外分支（对称结构） |
| `attention` | [B,96,H,W]→[B,96,H,W] | CBAM 注意力（通道+空间） |
| `decode4` | [B,96,H,W]→[B,64,H,W] | 膨胀卷积解码器 (dilation=3) |
| `decode3` | [B,64,H,W]→[B,32,H,W] | |
| `decode2` | [B,32,H,W]→[B,16,H,W] | |
| `decode1` | [B,16,H,W]→[B,1,H,W] | Tanh/2+0.5 输出 |

### `BiSeNet`（`network/SegNet.py`）

| 属性 | 说明 |
|------|------|
| `self.cp` | ContextPath，含 ResNet-18 骨干 + ARM 注意力精炼 + 自顶向下特征融合 |
| `self.conv_out` | 主输出头：BiSeNetOutput(128, 128, n_classes) |
| `self.conv_out16` | 辅助输出头：BiSeNetOutput(128, 64, n_classes) |
| `forward(x)` | 返回 `(feat_out, feat_out16)`，均为 `[B, n_classes, H, W]` |

### `DinoGuidance`（`network/DinoGuidance.py`）【新增优化模块】

| 属性/方法 | 说明 |
|-----------|------|
| `self.dino` | 冻结的 `dinov2_vits14` 模型 |
| `self.mean/self.std` | ImageNet 归一化参数 [0.485, 0.456, 0.406] / [0.229, 0.224, 0.225] |
| `preprocess(image)` | [B,1/3,H,W]→[B,3,224,224]，插值+归一化 |
| `forward(image)` | 提取融合图特征（保留梯度），返回 [B, 384] |
| `extract_target_feature(image)` | 提取可见光特征（截断梯度），返回 [B, 384] |

---

## 6.2 损失函数

| 类 | 文件 | 用途 | 关键参数 |
|----|------|------|----------|
| `Taylor_loss` | loss/Taylor.py | Stage 1 泰勒重建损失 | `L1 + grad + 0.3*high_freq` |
| `Fusionloss` | loss/Fusion.py | Stage 2 融合损失 | `L1_in + 10*L1_grad` |
| `FusionLoss` | loss/Task.py | Stage 3 融合损失 | `L1_in + 10*L1_grad` |
| `OhemCELoss` | loss/Task.py | 在线困难样本挖掘 CE | `thresh=0.7, n_min=307199` |
| `DinoSemanticLoss` | loss/DinoLoss.py | DINOv2 语义一致性 | `1-cos_sim, lambda=0.01` 【新增】|

---

## 6.3 数据集类

| 类 | 文件 | 输出 |
|----|------|------|
| `TaylorDataset` | Datasets.py | `img [1,H,W]` |
| `FusionDataset` | Datasets.py | `(ir [1,H,W], vis [1,H,W])` |
| `Fusion_dataset` | Datasets.py | `(vis [3,H,W], ir [1,H,W], label [H,W], name)` |

---

## 6.4 训练脚本关键变量

| 变量 | 脚本 | 含义 |
|------|------|------|
| `layer` | 全部 | 泰勒展开阶数，默认 2（即 y0+y1+y2） |
| `num = (epoch//10)+1` | train_task.py:495 | 分割 Loss 渐进权重 |
| `lambda_dino` | train_task.py | DINO Loss 权重，默认 0.01 |
| `score_thres=0.7` | train_task.py:414 | OHEM 阈值 |
| `n_min=640*480-1` | train_task.py:416 | OHEM 最小样本数 |

---

## 6.5 工具函数

| 函数 | 文件 | 说明 |
|------|------|------|
| `tensor_rgb2ycbcr` | utils.py | RGB→YCbCr，输出 Y/Cb/Cr 各 `[B,1,H,W]` |
| `tensor_ycbcr2rgb` | utils.py | YCbCr→RGB，拼接为 `[B,3,H,W]` |
| `normalize1` | utils.py | min-max 归一化到 [0,1] |
| `save_PIL` | utils.py | tensor→PIL→保存 |
| `transform_img` | utils.py | PIL→tensor `[1,1,H,W]` |
| `rgb2ycrcb` | train_task.py | RGB→YCrCb（注意：Cr/Cb 顺序与 utils.py 不同） |
| `ycrcb2rgb` | train_task.py | YCrCb→RGB |
| `generate_fusion` | train_task.py | 封装 Taylor+Fuse+色彩空间转换的完整前向 |
