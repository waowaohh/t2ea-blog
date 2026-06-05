---
title: "训练与推理流程"
date: 2025-06-05
draft: false
description: "数据预处理、三阶段训练逻辑（Taylor→Fusion→Task）、训练编排器与测试推理的完整流程"
tags: ["训练流程", "推理", "数据预处理", "Pipeline"]
categories: ["训练与推理"]
weight: 5
---

## 5.1 数据预处理

**Stage 1（TaylorDataset）**：
- 输入：灰度图像，`transforms.ToTensor()` + `transforms.Grayscale()`
- 输出：`[1, H, W]`，值域 `[0, 1]`
- 无数据增强

**Stage 2（FusionDataset）**：
- IR：灰度，`[1, H, W]`，`[0, 1]`
- VIS：RGB→灰度，`[1, H, W]`，`[0, 1]`
- 无数据增强

**Stage 3（Fusion_dataset）**：
- VIS：RGB `[3, H, W]`，`/255.0`
- IR：灰度 `[1, H, W]`，`/255.0`
- Label：`[H, W]`，int64（9 类语义标签）
- 训练时在 `train_task.py` 中做 RGB→YCbCr 转换，只融合 Y 通道

---

## 5.2 训练逻辑：分阶段训练

**不是端到端训练**，而是三阶段妥协训练：

1. **Stage 1**（200 epochs）：单独训练 Taylor_Encoder
   - 优化器：Adam, lr=1e-5
   - 目标：学会将图像分解为泰勒级数项

2. **Stage 2**（200 epochs）：冻结 Taylor_Encoder，训练 FusionModel
   - 优化器：Adam, lr=1e-5
   - 支持 DDP 多卡
   - 目标：学会将多阶泰勒特征融合为高质量图像

3. **Stage 3**（50 epochs）：冻结 Taylor_Encoder + BiSeNet，微调 FusionModel
   - 优化器：Adam, lr=1e-5
   - 加载 Stage 2 的 FusionModel 权重
   - 可选 DINOv2 引导
   - 目标：在保持融合质量的同时提升语义分割精度

**训练编排**（`train_pipeline.py`）：
- 创建统一的 timestamped run 目录
- 每个 stage 的 checkpoint 传递给下一个 stage
- 支持 `--start-stage`/`--stop-stage` 控制执行范围
- 支持 `--dry-run` 只生成命令不执行

---

## 5.3 测试推理

**文件**：`test/test_image.py`

```python
def fusion_test(vis, ir, save_path, net, fusion, device, filename, layer):
    if channels == 3:
        gray, img_cb, img_cr = tensor_rgb2ycbcr(vis)    # RGB→YCbCr
        _, y_vis = net(gray, layer)                       # Taylor 分解 Y 通道
        _, y_ir = net(ir, layer)                          # Taylor 分解 IR
        result = fusion(y_vis, y_ir)                      # 融合 Y 通道
        result = tensor_ycbcr2rgb(result, img_cb, img_cr) # YCbCr→RGB
    else:
        _, y_vis = net(vis, layer)
        _, y_ir = net(ir, layer)
        result = fusion(y_vis, y_ir)
```

**推理流程**：
1. 加载 `Taylor_Encoder` 和 `FusionModel` 权重
2. 对每对 IR+VIS 图像：
   - VIS 如果是 RGB，转换到 YCbCr，只融合 Y 通道
   - Taylor 分解 → 融合 → 逆泰勒重建
   - 用原始 Cb/Cr 通道恢复色彩
3. 保存结果图像
