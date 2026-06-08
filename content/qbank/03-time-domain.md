---
title: "自动控制原理题库：时域分析"
date: 2025-06-05
draft: false
math: true
description: "一阶/二阶系统响应、稳态误差、动态性能指标、主导极点等时域分析核心题目含答案与解析"
tags: ["自动控制原理", "时域分析", "稳态误差", "二阶系统", "题库"]
categories: ["自动控制原理题库"]
weight: 13
---

> 本章涵盖二阶系统的阶跃响应、时域指标及其系统型别，稳态误差与稳定性分析等。所有题目均配有详细解析。

### Q6/66：设某单位反馈二阶系统 \(G(s)=1/(s(s+2))\)，简述其闭环特征根分布及阶跃响应波形。

**🔑 答案**：

闭环传递函数：\(\Phi(s) = \frac{1}{s^2+2s+1} = \frac{1}{(s+1)^2}\)

**特征根**：\(s_{1,2} = -1\)（两个相等的负实根，在负实轴上重叠）。

**阶跃响应波形**：属于**临界阻尼**状态——无超调、单调上升曲线，最终稳态值为 1。是最快的无超调响应。

与标准欠阻尼波形（典型二阶阶跃响应）对比：欠阻尼时 \(\zeta=0.707\) 对应的特征根为 \(s_{1,2} = -\zeta\omega_n \pm j\omega_n\sqrt{1-\zeta^2}\)，会产生衰减振荡，超调量约为 4.3%。

---

### Q17/51/77/111：描述系统时域动态性能的指标有哪些？哪些评价响应速度？哪些评价阻尼？

**🔑 答案**：

<table>
<thead><tr>
<th style="text-align:left">指标</th>
<th style="text-align:left">定义</th>
<th style="text-align:left">类型</th>
</tr></thead><tbody>
<tr>
<td style="text-align:left">上升时间 \(t_r\)</td>
<td style="text-align:left">响应从终值10%→90%时间</td>
<td style="text-align:left">速度</td>
</tr>
<tr>
<td style="text-align:left">峰值时间 \(t_p\)</td>
<td style="text-align:left">响应达到第一个峰值的时间</td>
<td style="text-align:left">速度</td>
</tr>
<tr>
<td style="text-align:left">超调量 \(\sigma\%\)</td>
<td style="text-align:left">\(\frac{y_{\max}-y_{\infty}}{y_{\infty}}\times100\%\)</td>
<td style="text-align:left">阻尼</td>
</tr>
<tr>
<td style="text-align:left">调节时间 \(t_s\)</td>
<td style="text-align:left">进入终值±2%误差带时间</td>
<td style="text-align:left">综合</td>
</tr>
</tbody></table>

- **快速性**由 \(t_r, t_p\) 评价
- **阻尼程度**由 \(\sigma\%\) 评价
- \(t_s\) 是综合性指标

超调量与阻尼比的定量关系（仅适用于二阶欠阻尼系统）：
<div>
$$\sigma\% = e^{-\pi\zeta/\sqrt{1-\zeta^2}} \times 100\%$$
</div>

---

### Q20/50/52/80/110/112：开环增益和系统型别对稳态误差及稳定性的影响。

**🔑 答案**：

**开环增益 \(K\) 的影响**：

<table>
<thead><tr>
<th style="text-align:left">\(K\) 增大</th>
<th style="text-align:left">优点</th>
<th style="text-align:left">缺点</th>
</tr></thead><tbody>
<tr>
<td style="text-align:left">稳态误差 \(e_{ss}\)</td>
<td style="text-align:left">减小</td>
<td style="text-align:left">—</td>
</tr>
<tr>
<td style="text-align:left">稳定性裕度</td>
<td style="text-align:left">—</td>
<td style="text-align:left">降低（相位裕度减小）</td>
</tr>
<tr>
<td style="text-align:left">动态响应</td>
<td style="text-align:left">响应加快</td>
<td style="text-align:left">超调量增大，可能振荡</td>
</tr>
</tbody></table>

**系统型别 \(v\) 的影响**：增加积分环节 \(1/s\)——
- 提高稳态精度（能跟踪更高阶输入信号）
- **但每增加一个积分环节，引入 \(-90^\circ\) 相角滞后**，严重降低稳定性裕度
- 工程上很少使用 \(v\geq3\) 的系统

**核心矛盾**：精度 vs 稳定性。这是控制系统设计的基本权衡。

---

### Q22/43/82/103：一阶系统 \(\Phi_1=1/(s+1)\)，\(\Phi_2=1/(2s+1)\)，分析稳定性。

**🔑 答案**：

两个系统均稳定。

- \(\Phi_1\)：特征根 \(s_1=-1\)，时间常数 \(\tau=1\)，稳定
- \(\Phi_2\)：特征根 \(s_2=-0.5\)，时间常数 \(\tau=2\)，稳定

线性定常系统稳定的充要条件：**所有闭环特征根均具有负实部**。

---

### Q33/93：典型二阶系统超调量跟什么有关？超调量越小则调节时间如何变化？

**🔑 答案**：

超调量 \(\sigma\%\) **仅与阻尼比 \(\zeta\) 有关**（见公式）。\(\zeta\) 越大，\(\sigma\%\) 越小。

**\(\zeta \uparrow\) → \(\sigma\% \downarrow\)**，调节时间变化取决于 \(\zeta\) 的范围：
- \(\zeta < 0.69\)：\(\zeta\uparrow\) → \(t_s\downarrow\)（超调变小，调节更快）
- \(\zeta > 0.69\)：\(\zeta\uparrow\) → \(t_s\uparrow\)（单调响应，但响应变慢）

工程最佳阻尼比：\(\zeta = 0.707\)（\(\sigma\% \approx 4.3\%\)，\(t_s\) 最小）。

---

### Q23/83：决定高阶系统动态性能的主要因素是什么？为什么？

**🔑 答案**：**闭环主导极点**。

**条件**：
1. 与虚轴距离最近（实部绝对值最小）
2. 其他极点实部绝对值 ≥ 5×主导极点实部绝对值
3. 不受相邻闭环零点显著影响（零点对消）

满足上述条件的高阶系统可**降阶**为二阶系统，用二阶系统的方法分析其动态性能。

---

### 本章小结

时域分析的核心是二阶系统模型：阻尼比 \(\zeta\) 决定振荡特性，自然频率 \(\omega_n\) 决定响应速度。系统型别 \(v\) 和开环增益 \(K\) 共同决定稳态误差，"稳、准、快"三者的平衡是控制系统设计的永恒主题。