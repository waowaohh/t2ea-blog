---
title: "损失函数彻底拆解"
date: 2025-06-05
draft: false
math: true
description: "Stage 1 Taylor Loss、Stage 2 Fusion Loss、Stage 3 三重 Loss 联合优化的完整数学推导与代码映射"
tags: ["损失函数", "Taylor Loss", "Fusion Loss", "OHEM", "DINO Loss"]
categories: ["损失函数"]
weight: 4
---

## 4.1 Stage 1: Taylor Loss (`loss/Taylor.py`)

```python
class Taylor_loss:
    def forward(self, input, result, gd):
        loss_L1 = F.l1_loss(input, result)                    # 强度损失
        loss_g1 = F.l1_loss(Sobel(input), Sobel(result))     # 梯度损失
        grad_map = max(gd[1], gd[2], ..., gd[n])             # 逐阶取最大梯度
        loss_g2 = F.l1_loss(Sobel(input), grad_map)           # 高频残差约束
        loss_g = loss_g1 + 0.3 * loss_g2
        return loss_L1, loss_g, loss_L1 + loss_g
```

**数学公式**：

<div>
$$\mathcal{L}_{Taylor} = \underbrace{\|x - \hat{x}\|_1}_{\text{重建强度}} + \underbrace{\|\nabla x - \nabla \hat{x}\|_1}_{\text{重建梯度}} + 0.3 \cdot \underbrace{\|\nabla x - \max_i(g_i)\|_1}_{\text{高频残差约束}}$$
</div>

**作用**：
- \(\mathcal{L}\_{int}\)：确保泰勒级数能重建原始图像
- \(\mathcal{L}\_{grad}\)：确保边缘/纹理被保留
- \(0.3 \cdot \mathcal{L}\_{g2}\)：约束高阶梯度项 \(y\_1, y\_2, \ldots\) 的最大值应覆盖输入的梯度信息，确保高频分量被充分编码

---

## 4.2 Stage 2: Fusion Loss (`loss/Fusion.py`)

```python
class Fusionloss:
    def forward(self, image_vis, image_ir, generate_img):
        image_y = image_vis[:, :1]                              # 可见光亮度通道
        x_in_max = torch.max(image_y, image_ir)                 # 逐像素取最亮
        loss_in = F.l1_loss(x_in_max, generate_img)             # 强度损失

        y_grad = Sobel(image_y)
        ir_grad = Sobel(image_ir)
        x_grad_joint = torch.max(y_grad, ir_grad)               # 逐像素取最大梯度
        loss_grad = F.l1_loss(Sobel(generate_img), x_grad_joint) # 梯度损失

        loss_total = loss_in + 10 * loss_grad
```

**数学公式**：

<div>
$$\mathcal{L}_{fusion} = \underbrace{\|\max(Y_{vis}, I_{IR}) - I_F\|_1}_{\text{强度L1}} + 10 \cdot \underbrace{\|\max(\nabla Y_{vis}, \nabla I_{IR}) - \nabla I_F\|_1}_{\text{梯度L1}}$$
</div>

**作用**：
- 强度损失：融合图应保留两幅输入中的最亮像素（红外热目标通常是高亮的）
- 梯度损失（权重 10）：融合图的边缘应覆盖两幅输入中最强的边缘信息
- 权重 10 倍：梯度/边缘信息比强度信息更重要，这符合融合任务的感知需求

---

## 4.3 Stage 3: 三重 Loss 联合优化

### 4.3.1 FusionLoss (`loss/Task.py`)

```python
class FusionLoss:  # 与 Stage 2 的 Fusionloss 逻辑相同
    def forward(self, image_A, image_B, image_F):
        loss_int = F.l1_loss(image_F, torch.max(image_A, image_B))
        loss_grad = F.l1_loss(Sobel(image_F), torch.max(Sobel(image_A), Sobel(image_B)))
        return loss_int, loss_grad, loss_int + 10 * loss_grad
```

### 4.3.2 OhemCELoss（在线困难样本挖掘交叉熵）

```python
class OhemCELoss:
    def forward(self, logits, labels):
        loss = CrossEntropy(logits, labels, reduction='none').view(-1)  # 逐像素CE
        loss, _ = torch.sort(loss, descending=True)                     # 降序排列
        n_min = min(self.n_min, max(loss.numel()-1, 0))
        if loss[n_min] > self.thresh:
            loss = loss[loss > self.thresh]     # 只保留 loss > 阈值的困难样本
        else:
            loss = loss[:max(n_min, 1)]         # 至少保留 n_min 个样本
        return torch.mean(loss)
```

**数学公式**：

<div>
$$\mathcal{L}_{OHEM} = \frac{1}{|\mathcal{S}|} \sum_{i \in \mathcal{S}} \text{CE}(p_i, y_i)$$
</div>

其中 \(\mathcal{S}\) 是通过 OHEM 策略选出的困难样本集合：
- 若 \(L\_{(n_{min})} > \tau\)：\(\mathcal{S} = \{i : L_i > \tau\}\)
- 否则：\(\mathcal{S} = \{L_{(0)}, L_{(1)}, \ldots, L_{(n_{min})}\}\)

**参数**：`thresh=0.7`（对应 \(-\log(0.7) \approx 0.357\)），`n_min = 640*480-1 = 307199`。

### 4.3.3 总 Loss 构成

```python
# train_task.py:522-537
seg_loss = criteria_p(out, lb) + 0.1 * criteria_16(mid, lb)   # 主输出 + 0.1*辅助输出
loss_fusion = FusionLoss(vis_y, ir, logits)
num = (epoch // 10) + 1                                         # 渐进权重

loss_total = loss_fusion + num * seg_loss
if use_dino:
    dino_loss = 1 - cos(DINO(fusion_image), DINO(vis_image).detach())
    loss_total = loss_total + lambda_dino * dino_loss
```

**完整数学公式**：

<div>
$$\boxed{\mathcal{L}_{total} = \underbrace{\mathcal{L}_{int} + 10\mathcal{L}_{grad}}_{\mathcal{L}_{fusion}} + \underbrace{(\lfloor e/10 \rfloor + 1)}_{\text{渐进权重}} \cdot \underbrace{(\mathcal{L}_{OHEM}^{out} + 0.1\mathcal{L}_{OHEM}^{mid})}_{\mathcal{L}_{seg}} + \underbrace{\lambda_{dino}(1 - \cos(\mathbf{z}_f, \mathbf{z}_v))}_{\mathcal{L}_{dino}}}$$
</div>

---

## 4.4 Loss 平衡机制分析

<table>
<thead><tr>
<th style="text-align:left">Loss</th>
<th style="text-align:left">权重</th>
<th style="text-align:left">量级估计</th>
<th style="text-align:left">作用</th>
</tr></thead><tbody>
<tr>
<td style="text-align:left">\(\mathcal{L}\_{int}\)</td>
<td style="text-align:left">1</td>
<td style="text-align:left">~0.01-0.05</td>
<td style="text-align:left">保留最亮像素（红外目标+可见光细节）</td>
</tr>
<tr>
<td style="text-align:left">\(\mathcal{L}\_{grad}\)</td>
<td style="text-align:left">10</td>
<td style="text-align:left">~0.001-0.01</td>
<td style="text-align:left">保留边缘结构</td>
</tr>
<tr>
<td style="text-align:left">\(\mathcal{L}\_{seg}^{out}\)</td>
<td style="text-align:left">num (1→6)</td>
<td style="text-align:left">~0.5-2.0</td>
<td style="text-align:left">分割精度引导融合质量</td>
</tr>
<tr>
<td style="text-align:left">\(\mathcal{L}\_{seg}^{mid}\)</td>
<td style="text-align:left">0.1×num</td>
<td style="text-align:left">~0.05-0.2</td>
<td style="text-align:left">辅助分割监督</td>
</tr>
<tr>
<td style="text-align:left">\(\mathcal{L}\_{dino}\)</td>
<td style="text-align:left">0.01</td>
<td style="text-align:left">~0.001-0.01</td>
<td style="text-align:left">语义一致性正则</td>
</tr>
</tbody></table>

**平衡策略**：
1. **梯度 Loss 权重 10**：因为 Sobel 梯度值通常比像素值小一个量级，乘以 10 使其与强度 Loss 在同一量级
2. **渐进语义权重**：`num = epoch//10 + 1`，前 10 epoch 以融合为主（num=1），后续逐步加强语义约束
3. **DINO 权重 0.01**：作为轻量正则项，避免 DINO 语义约束过度干扰融合质量
