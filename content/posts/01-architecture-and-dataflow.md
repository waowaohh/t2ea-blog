---
title: "项目整体架构与优化数据流"
date: 2025-06-05
draft: false
math: true
description: "T2EA 项目的文件调用关系、端到端数据流及 DINOv2 优化切入点定位"
tags: ["架构", "数据流", "DINOv2"]
categories: ["技术架构"]
weight: 1
---

> 本系列基于当前最新代码（codex/t2ea-training-and-eval-updates 分支）的完整剖析。

## 1.1 文件调用关系图

```
train_pipeline.py（编排器）
    ├── Stage 1: train_TEM.py → network/TEM.py (Taylor_Encoder)
    │                   ├── loss/Taylor.py (Taylor_loss)
    │                   └── Datasets.py (TaylorDataset)
    ├── Stage 2: train_Fusion.py → network/FusionNet.py (FusionModel)
    │                   ├── loss/Fusion.py (Fusionloss)
    │                   ├── Datasets.py (FusionDataset)
    │                   └── 冻结 Taylor_Encoder
    └── Stage 3: train_task.py → network/FusionNet.py (FusionModel)
                        ├── network/SegNet.py (BiSeNet) [冻结]
                        ├── network/DinoGuidance.py (DinoGuidance) [冻结]
                        ├── loss/Task.py (FusionLoss + OhemCELoss)
                        ├── loss/DinoLoss.py (DinoSemanticLoss)
                        ├── Datasets.py (Fusion_dataset, 带label)
                        └── 冻结 Taylor_Encoder

推理:
    test/test_image.py
        ├── network/TEM.py (Taylor_Encoder)
        ├── network/FusionNet.py (FusionModel)
        └── utils.py (色彩空间转换)

评估:
    test/eval_fusion.py → 9项融合质量指标
    test/compare_eval.py → 多轮对比分析
```

## 1.2 端到端数据流（含 DINO 优化）

```
输入: IR [B,1,H,W] + VIS [B,3,H,W]
        │
        ▼
  ┌─ rgb2ycbcr(VIS) ─→ Y [B,1,H,W], Cb, Cr
  │
  ├─ Taylor_Encoder(Y, n=2) ─→ y_list_vis = [y0, y1, y2]
  │      y0 = base(Y)                    # 零阶项（低频语义）
  │      y1 = gradient(cat(y0, Y))       # 一阶梯度项
  │      y2 = gradient(cat(y1, Y))       # 二阶梯度项
  │      recon = y0 + y1 + y2/2
  │
  ├─ Taylor_Encoder(IR, n=2) ─→ y_list_ir = [y0, y1, y2]
  │
  ▼
  FusionModel(y_list_ir, y_list_vis)
      │
      ├─ 对每层 i: FusionNetwork(ir[i], vis[i])
      │    ├─ vis: Conv→RGBD1→RGBD2 → vis_feat [B,48,H,W]
      │    ├─ ir:  Conv→RGBD1→RGBD2 → ir_feat  [B,48,H,W]
      │    ├─ CBAM Attention(cat(vis_feat, ir_feat))
      │    └─ Dilated Decoder → fused_i [B,1,H,W]
      │
      └─ 逆泰勒重建: result = Σ fused_i / i!
          │
          ▼
  ycrcb2rgb(result, Cb, Cr) → 融合RGB图像 [B,3,H,W]
          │
          ▼
  ┌─ BiSeNet(fusion_image) → seg_logits → OhemCELoss(seg, label)
  │
  └─ DinoGuidance(fusion_image) vs DinoGuidance(vis) → DinoSemanticLoss
          │
          ▼
  L_total = L_fusion + num * L_seg + λ_dino * L_dino
```

## 1.3 新优化切入点定位

**DINOv2 语义引导** 是当前最新的优化，切入位置在 Stage 3 的 **融合后语义一致性约束** 阶段：

- **上游影响**：融合图像 `fusion_image` 需要经过 `torch.clamp(0,1)` 后送入 DINOv2，梯度通过 `fusion_image` 反传到 `FusionModel`
- **下游影响**：DINOv2 的梯度只流经 `FusionModel`，不流经冻结的 Taylor_Encoder 和 BiSeNet
- **关键设计**：`visible_feature` 使用 `.detach()` 切断梯度，`dino_teacher` 参数 `requires_grad_(False)` 完全冻结
