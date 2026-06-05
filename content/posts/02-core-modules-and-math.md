---
title: "核心模块与数学原理的源码映射"
date: 2025-06-05
draft: false
math: true
description: "泰勒展开近似网络、双分支特征融合网络、BiSeNet 语义分割网络与 DINOv2 语义引导模块的源码级解析"
tags: ["泰勒展开", "FusionNetwork", "BiSeNet", "DINOv2", "数学原理"]
categories: ["核心模块"]
weight: 2
---

## 2.1 泰勒展开近似网络（Taylor_Encoder）

**数学原理**：

$$
f(x) \approx f(x_0) + f'(x_0)(x-x_0) + \frac{f''(x_0)}{2!}(x-x_0)^2 + \cdots
$$

代码中将输入图像 $x$ 在 $x_0=0$ 处展开：

$$
\hat{x} = y_0 + y_1 + \frac{y_2}{2!} + \frac{y_3}{3!} + \cdots
$$

**源码映射**（`network/TEM.py:57-73`）：

```python
class Taylor_Encoder(nn.Module):
    def forward(self, input, n):
        y_list = []
        x = self.base(input)           # y_0 = f(x_0)，零阶项（低频/语义主导项）
        y_list.append(x)
        for i in range(1, n+1):
            # y_i = ∂ⁱf/∂xⁱ，高阶梯度项（高频残差项）
            # 输入: cat(y_{i-1}, input) —— 前一层输出与原始输入拼接
            y_list.append(self.gradient(torch.cat([y_list[i-1], input], dim=1)))
        for i in range(len(y_list)):
            result += (1 / factorial(i)) * y_list[i]  # 泰勒级数求和
        return result, y_list
```

**关键 Tensor 变换**：

| 变量 | Shape | 含义 |
|------|-------|------|
| `input` | `[B,1,H,W]` | 灰度输入 |
| `y_list[0]` = `self.base(input)` | `[B,1,H,W]` | 零阶项：Conv(1→32)→ResB(32→64)→ResB(64→32)→ResB(32→1) |
| `y_list[i]` = `self.gradient(cat(y_{i-1}, input))` | `[B,1,H,W]` | 第 i 阶项：Conv(2→8)→RDBLOCK(8→16)→RDBLOCK(16→32)→RDBLOCK(32→64)→Conv(64→1) |
| `result` | `[B,1,H,W]` | 泰勒级数重建结果 |

**低频 vs 高频对应**：
- `self.base`（ResB 块）：感受野大、通道压缩→恢复，提取低频语义信息
- `self.gradient`（RDBLOCK 密集连接块）：输入包含前一层输出+原始输入，密集连接捕获逐阶残差/高频细节

---

## 2.2 双分支特征融合网络（FusionNetwork）

**源码**（`network/FusionNet.py:106-143`）：

```python
class FusionNetwork(nn.Module):
    def forward(self, image_vis, image_ir):
        # 双分支编码器
        x_vis_p  = self.vis_conv(image_vis[:, :1])   # [B,1,H,W] → [B,16,H,W]
        x_vis_p1 = self.vis_rgbd1(x_vis_p)           # [B,16,H,W] → [B,32,H,W]
        x_vis_p2 = self.vis_rgbd2(x_vis_p1)          # [B,32,H,W] → [B,48,H,W]

        x_inf_p  = self.inf_conv(image_ir)            # [B,1,H,W] → [B,16,H,W]
        x_inf_p1 = self.inf_rgbd1(x_inf_p)            # [B,16,H,W] → [B,32,H,W]
        x_inf_p2 = self.inf_rgbd2(x_inf_p1)           # [B,32,H,W] → [B,48,H,W]

        # CBAM 注意力融合
        x_attention = self.attention(cat(x_vis_p2, x_inf_p2))  # [B,96,H,W] → [B,96,H,W]

        # 膨胀卷积解码器
        x = self.decode4(x_attention)   # [B,96,H,W] → [B,64,H,W], dilation=3
        x = self.decode3(x)             # [B,64,H,W] → [B,32,H,W], dilation=3
        x = self.decode2(x)             # [B,32,H,W] → [B,16,H,W], dilation=3
        x = self.decode1(x)             # [B,16,H,W] → [B,1,H,W], tanh/2+0.5
        return x
```

**融合策略详解**：

1. **RGBD 模块**（`RGBD` 类）：每个分支内部使用 DenseBlock（密集连接）+ Sobelxy（梯度边缘）并行提取特征后相加
2. **CBAM 注意力**（`cbam_block` 类）：Channel Attention（双池化→FC→Sigmoid） × Spatial Attention（均值+最大→Conv→Sigmoid）
3. **解码器**：使用 dilation=3 的膨胀卷积扩大感受野，逐步通道缩减到 1

**逆泰勒重建**（`FusionModel.forward`）：

```python
class FusionModel(nn.Module):
    def forward(self, ir, vis):
        result = torch.zeros([b, c, h, w], device=device)
        for i in range(n):
            fused = self.Net(ir[i], vis[i])  # 每层独立融合
            result += (1 / factorial(i)) * fused  # 逆泰勒加权求和
        return result
```

数学含义：对泰勒分解的每一阶特征分别融合，再以 $1/i!$ 权重重建，保持与泰勒展开的一致性。

---

## 2.3 BiSeNet 语义分割网络

**架构**（`network/SegNet.py`）：

- **骨干网络** `Resnet18`：加载预训练 ResNet-18 权重，输出 4 级特征 `[feat4, feat8, feat16, feat32]`（1/4, 1/8, 1/16, 1/32 下采样）
- **上下文路径** `ContextPath`：
  - 对 feat8/feat16/feat32 分别使用 `AttentionRefinementModule`（全局池化→1x1 Conv→Sigmoid 通道注意力）
  - 自上而下融合：feat32→upsample→cat(feat16)→sp16→upsample→cat(feat8)→sp8
  - `conv_fuse1/conv_fuse2`：ConvBNSig（Sigmoid 注意力门控）逐级加权
- **输出头**：
  - `conv_out`：主输出，128→128→n_classes
  - `conv_out16`：辅助输出，128→64→n_classes
  - 两个输出都 bilinear upsample 到原始分辨率

**前向传播**（`BiSeNet.forward`）：

```python
def forward(self, x):  # x: [B,3,H,W]
    feat_res8, feat_cp8, feat_cp16 = self.cp(x)
    feat_out = self.conv_out(feat_res8)        # 主分支 [B,n_cls,H/8,W/8]
    feat_out16 = self.conv_out16(feat_cp8)     # 辅助分支 [B,n_cls,H/8,W/8]
    feat_out = F.interpolate(feat_out, (H,W))  # 上采样到原尺寸
    feat_out16 = F.interpolate(feat_out16, (H,W))
    return feat_out, feat_out16  # 双输出用于 OHEM Loss
```

---

## 2.4 DINOv2 语义引导模块

**架构**（`network/DinoGuidance.py`）：

```python
class DinoGuidance(nn.Module):
    def __init__(self, model_name="dinov2_vits14", ...):
        self.dino = self._load_model()  # 冻结的 ViT-S/14
        self.dino.eval()
        for param in self.dino.parameters():
            param.requires_grad_(False)

    def preprocess(self, image):  # [B,1/3,H,W] → [B,3,224,224] ImageNet归一化
        if channels == 1:
            image = image.repeat(1, 3, 1, 1)
        image = F.interpolate(image, size=(224,224), mode="bilinear")
        return (image - self.mean) / self.std

    def forward(self, image):  # 保留梯度，用于融合图像
        image = self.preprocess(image)
        return self._select_feature(self.dino.forward_features(image))

    def extract_target_feature(self, image):  # 切断梯度，用于可见光
        with torch.no_grad():
            feature = self.forward(image)
        return feature.detach()
```

**特征提取** `_select_feature`：支持 dict 格式（优先 `x_norm_clstoken`）或 tensor 格式（2D→直接返回，3D→mean pooling）。

**DINOv2 Loss**（`loss/DinoLoss.py`）：

$$
\mathcal{L}_{dino} = 1 - \cos(\mathbf{z}_f, \mathbf{z}_v)
$$

其中 $\mathbf{z}_f = \Phi_{DINOv2}(I_f)$（融合图特征，保留梯度），$\mathbf{z}_v = \text{stopgrad}(\Phi_{DINOv2}(I_v))$（可见光特征，截断梯度）。
