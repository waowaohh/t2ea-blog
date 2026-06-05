---
title: "关键发现与建议"
date: 2025-06-05
draft: false
math: true
description: "T2EA 项目已确认的6项设计亮点与4项建议修复优先级汇总"
tags: ["设计亮点", "Bug修复", "优化建议"]
categories: ["总结"]
weight: 7
---

## 7.1 已确认的设计亮点

1. **泰勒展开的可解释性分解**：将图像分解为低频语义（base）+ 高频梯度（gradient），每一阶都有清晰的物理含义
2. **逆泰勒一致性**：编码和解码都使用 $1/i!$ 权重，保持数学一致性
3. **CBAM 注意力融合**：在通道和空间维度上自适应加权红外/可见光特征
4. **渐进式语义引导**：`num = epoch//10 + 1` 优雅地平衡融合质量和语义精度
5. **OHEM 困难样本挖掘**：分割网络关注难以分类的像素（通常是目标边界）
6. **DINOv2 语义正则**：利用预训练视觉 Transformer 的语义一致性作为额外约束

---

## 7.2 建议修复的优先级

| 优先级 | 问题 | 影响 |
|--------|------|------|
| **高** | loss/Taylor.py 中 Sobelxy 的 device 硬编码 | 多卡训练时 crash |
| **高** | train_TEM.py 中 loss 累积计算错误 | 训练日志 loss 值不准确，影响 best model 选择 |
| **中** | stage3 语义权重无上界 | num 可能过大导致 loss 不平衡 |
| **低** | FusionModel.forward 中 torch.zeros 风格 | 代码清晰度 |

---

*文档生成时间：基于 codex/t2ea-training-and-eval-updates 分支最新代码*
