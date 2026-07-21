# SFT（Supervised Fine-Tuning）基础理论

## 目录

- [1. SFT 在后训练流程中的位置](#1-sft-在后训练流程中的位置)
- [2. 训练数据长什么样](#2-训练数据长什么样)
  - [2.1 单轮、多轮与工具调用](#21-单轮多轮与工具调用)
- [3. SFT 的核心目标函数](#3-sft-的核心目标函数)
  - [3.1 符号与索引约定](#31-符号与索引约定)
  - [3.2 条件概率的自回归建模](#32-条件概率的自回归建模)
  - [3.3 最大似然与带掩码的交叉熵](#33-最大似然与带掩码的交叉熵)
  - [3.4 为什么需要 loss mask](#34-为什么需要-loss-mask)
  - [3.5 Teacher forcing](#35-teacher-forcing)
- [4. 一次训练迭代发生了什么](#4-一次训练迭代发生了什么)
- [5. 全参数微调、LoRA 与 QLoRA](#5-全参数微调lora-与-qlora)
- [6. 为什么数据质量通常比数据数量更重要](#6-为什么数据质量通常比数据数量更重要)
- [7. 关键训练选择](#7-关键训练选择)
  - [7.1 截断与样本长度](#71-截断与样本长度)
  - [7.2 token 平均与样本平均](#72-token-平均与样本平均)
  - [7.3 学习率、训练轮数与遗忘](#73-学习率训练轮数与遗忘)
- [8. 常见问题与排查方向](#8-常见问题与排查方向)
- [9. 如何评估一个 SFT 模型](#9-如何评估一个-sft-模型)
- [10. SFT 的能力边界](#10-sft-的能力边界)
- [11. 本章小结](#11-本章小结)
- [附录 A：为什么最大似然估计会得到 SFT 目标函数](#附录-a为什么最大似然估计会得到-sft-目标函数)
  - [A.1 建模目标与基本假设](#a1-建模目标与基本假设)
  - [A.2 从单条样本到整个训练集的似然](#a2-从单条样本到整个训练集的似然)
  - [A.3 为什么要取对数](#a3-为什么要取对数)
  - [A.4 使用链式法则展开到 token 级别](#a4-使用链式法则展开到-token-级别)
  - [A.5 从回答序列推广到 loss mask](#a5-从回答序列推广到-loss-mask)
  - [A.6 为什么负对数似然等于交叉熵](#a6-为什么负对数似然等于交叉熵)
  - [A.7 从 KL 散度理解为什么 MLE 合理](#a7-从-kl-散度理解为什么-mle-合理)
  - [A.8 推导成立不等于数据目标一定正确](#a8-推导成立不等于数据目标一定正确)
- [参考文献](#参考文献)

预训练通过“预测下一个 token”让大模型掌握语言规律、知识和文本续写能力，但基座模型并不一定能稳定地理解并遵循用户指令，也未必知道应以何种方式完成具体任务。监督微调（Supervised Fine-Tuning，SFT）要解决的核心问题，就是利用高质量的“指令—回答”示范，将模型的通用续写能力转化为更加稳定的指令遵循和任务执行能力。本章将解释 SFT 在做什么、为什么有效，以及实践中最容易出错的地方；暂不绑定某个训练框架。

## 1. SFT 在后训练流程中的位置

一个常见的大模型训练流程可以简化为：

```text
海量无标注文本 ──预训练──> 基座模型 ──SFT──> 指令模型 ──偏好优化──> 更符合人类偏好的模型
                                           （DPO、PPO、GRPO 等）
```

- **预训练（Pre-training）**：从海量文本中学习语言规律、知识和一定的推理能力，核心任务通常是“预测下一个 token”。
- **SFT**：使用人工编写、模型合成或业务沉淀的“输入—理想输出”样本，教模型按照指令完成任务。
- **偏好优化（Preference Optimization）**：进一步利用回答之间的偏好、奖励信号或可验证结果，让模型更有用、更安全或更符合特定目标。

三者并非完全割裂。SFT 仍然使用 next-token prediction，只是训练数据从通用文本变成了高质量示范。可以把它理解为：**预训练主要赋予知识和能力，SFT 主要教模型在什么场景下、以什么方式运用这些知识/能力。** Instruction tuning 通常就是以自然语言指令为数据形式的 SFT；聊天模型训练则是它的多轮对话版本。

> SFT 能补充领域知识，但不适合被简单理解成“向模型数据库里写入大量新事实”。若目标主要是让知识可更新、可追溯，检索增强生成（RAG）往往更合适；若目标是稳定地改变回答方式、任务能力或输出格式，SFT 更直接。

## 2. 训练数据长什么样

最简单的一条样本由输入 $x$ 和目标回答 $y$ 组成：

```json
{
  "instruction": "把下面句子翻译成英文：今天天气很好。",
  "response": "The weather is nice today."
}
```

聊天模型通常使用消息列表，以区分 system、user 和 assistant：

```json
{
  "messages": [
    {"role": "system", "content": "你是一名简洁、准确的翻译助手。"},
    {"role": "user", "content": "把下面句子翻译成英文：今天天气很好。"},
    {"role": "assistant", "content": "The weather is nice today."}
  ]
}
```

训练前，chat template 会把结构化消息转换成模型真正看到的 token 序列，例如：

```text
<system>你是一名简洁、准确的翻译助手。</system>
<user>把下面句子翻译成英文：今天天气很好。</user>
<assistant>The weather is nice today.</assistant><eos>
```

这里的标签只是示意。不同模型使用不同的特殊 token 和模板，必须以对应 tokenizer 的 chat template 为准。**训练与推理的模板不一致，会造成明显的分布偏移。**

> **TODO**：拿一个真实模型的 chat template 来看下给模型的真实输入是啥。

### 2.1 单轮、多轮与工具调用

- **单轮数据**适合问答、分类、抽取、翻译和固定格式生成等任务。
- **多轮数据**还要教会模型利用上下文、处理指代，并保持角色边界。
- **工具调用数据**需要包含工具描述、参数格式、工具返回结果及最终回答，用示范教会模型“何时调用、如何调用”。

多轮对话中，可以只监督最后一个 assistant 回答，也可以监督所有 assistant 回答。后者能利用更多训练信号，但必须正确屏蔽 system、user、工具返回值以及轮次分隔符中不希望模型学习预测的部分。

## 3. SFT 的核心目标函数

### 3.1 符号与索引约定

为便于后续章节保持一致，本教程统一使用以下符号：

| 符号 | 含义 |
| --- | --- |
| $\mathcal{D}$ | SFT 训练集 |
| $N$ | 训练集中的样本数量 |
| $i$ | 样本下标， $i\in\{1,\ldots,N\}$ |
| $x_i$ | 第 $i$ 条样本的输入上下文，即目标回答之前的语义内容；在聊天场景中可包含 system 消息、历史对话和当前用户指令，之后再由 chat template 加入特殊 token 并序列化 |
| $y_i$ | 第 $i$ 条样本的目标回答 token 序列；若把“生成后停止”也视为目标的一部分，则序列末尾应包含模板规定的结束 token（如 `<eos>`） |
| $T_i$ | 目标回答 $y_i$ 的 token 数量；不同样本的回答长度可以不同 |
| $t$ | token 位置下标；在回答序列中 $t\in\{1,\ldots,T_i\}$，在模板化的完整序列中 $t\in\{1,\ldots,L_i\}$ |
| $y_{i,t}$ | 第 $i$ 条样本目标回答中的第 $t$ 个 token |
| $y_{i,\lt t}$ | 第 $i$ 条样本中位于 $y_{i,t}$ 之前的回答 token，即 $(y_{i,1},\ldots,y_{i,t-1})$；当 $t=1$ 时为空序列 |
| $\theta$ | 模型中参与当前训练目标的全部参数；全参数微调时是全部模型参数，LoRA 训练时通常是可训练的适配器参数 |
| $p_\theta(\cdot\mid\cdot)$ | 参数为 $\theta$ 的模型给出的条件概率分布 |
| $z_i$ | 将 $x_i$ 和 $y_i$ 按 chat template 序列化后的完整 token 序列，记为 $(z_{i,1},\ldots,z_{i,L_i})$ |
| $L_i$ | 完整序列 $z_i$ 的 token 数量，通常包含输入、目标回答和模板中的特殊 token，但不包含为组 batch 而额外添加的 padding |
| $z_{i,t}$ | 完整序列 $z_i$ 中的第 $t$ 个 token |
| $z_{i,\lt t}$ | 完整序列中位于 $z_{i,t}$ 之前的所有 token，即 $(z_{i,1},\ldots,z_{i,t-1})$ |
| $m_{i,t}$ | 第 $i$ 条样本第 $t$ 个位置的 loss mask；取值为 1 时计算该位置的损失，取值为 0 时忽略该位置 |
| $\mathcal{L}_{\mathrm{SFT}}(\theta)$ | 在所有有效监督 token 上取平均的 SFT 负对数似然损失 |

这里的“token”是经过 tokenizer 切分后得到的模型基本处理单位，不一定对应一个完整的汉字或英文单词。训练集写作

$$
\mathcal{D}=\{(x_i,y_i)\}_{i=1}^{N},
$$

表示 $\mathcal{D}$ 由 $N$ 个“输入上下文—目标回答”样本对组成。

### 3.2 条件概率的自回归建模

给定输入上下文 $x_i$，我们希望模型为完整目标回答 $y_i=(y_{i,1},\ldots,y_{i,T_i})$ 赋予较高的条件概率。根据概率的链式法则，任意序列的条件联合概率都可以写成逐 token 条件概率的乘积：

$$
p_\theta(y_i\mid x_i)
=\prod_{t=1}^{T_i}p_\theta(y_{i,t}\mid x_i,y_{i,\lt t}),
$$

自回归语言模型的 next-token prediction 恰好与上述链式分解逐项对应：在第 $t$ 步，模型以输入上下文 $x_i$ 和回答前缀 $y_{i,\lt t}$ 为条件，预测下一个 token $y_{i,t}$；将各步的条件概率相乘，就得到完整回答 $y_i$ 的条件概率。

### 3.3 最大似然与带掩码的交叉熵

SFT 通常使用最大似然估计（Maximum Likelihood Estimation，MLE）：调整参数 $\theta$，使训练集中的目标回答在对应输入条件下具有尽可能高的概率。最大化对数似然，等价于最小化负对数似然；对于分类形式的 next-token prediction，这就是 token 级交叉熵损失。完整推导见[附录 A](#附录-a为什么最大似然估计会得到-sft-目标函数)。

实际训练时，模型接收的是经过 chat template 序列化的完整序列 $z_i$。为了只让指定位置参与监督，引入 loss mask $m_{i,t}$，得到

$$
\mathcal{L}_{\mathrm{SFT}}(\theta)
=-
\frac{
\displaystyle\sum_{i=1}^{N}\sum_{t=1}^{L_i}
m_{i,t}\log p_\theta(z_{i,t}\mid z_{i,\lt t})
}{
\displaystyle\sum_{i=1}^{N}\sum_{t=1}^{L_i}m_{i,t}
}.
$$

式中各部分的含义如下：

- $p_\theta(z_{i,t}\mid z_{i,\lt t})$ 是模型根据完整序列前缀 $z_{i,\lt t}$，为真实 token $z_{i,t}$ 分配的条件概率；
- $\log$ 表示自然对数；概率越接近 1，对应的负对数损失越小；
- $m_{i,t}\in\{0,1\}$ 决定位置 $(i,t)$ 是否参与监督；乘以 0 相当于屏蔽该位置；
- 两个连加符号先遍历样本下标 $i=1,\ldots,N$，再遍历该样本的位置下标 $t=1,\ldots,L_i$；分子累加所有有效位置的 token 对数似然，分母 $\sum_{i=1}^{N}\sum_{t=1}^{L_i}m_{i,t}$ 是训练集中的有效监督 token 总数；
- 最外层负号把“最大化对数似然”转换为“最小化损失”。

训练的目标可以简写为

$$
\theta^*=\underset{\theta}{\arg\min}\ \mathcal{L}_{\mathrm{SFT}}(\theta).
$$

其中， $\arg\min$ 表示寻找使损失最小的参数， $\theta^*$ 表示在该训练目标下得到的最优参数。这里的“最优”是相对于给定数据、模型和优化过程而言的，实际训练通常只能得到近似解。

### 3.4 为什么需要 loss mask

以单轮样本为例：

| token 区域 | `<user>` 与用户问题 | `<assistant>` | 助手回答 | `<eos>` | padding |
| --- | --- | --- | --- | --- | --- |
| 常见 loss mask | 0 | 0 或 1 | 1 | 1 | 0 |

为组建 batch 而补齐的 padding 不属于原始序列 $z_i$，因此没有计入长度 $L_i$。在代码的定长张量中，这些额外位置仍然存在，但其 mask 必须为 0，labels 通常设为 `-100`。

这种做法常称为 **completion-only loss** 或 **response-only loss**：输入负责提供条件，主要让助手回答承担损失。`<assistant>` 是否计入损失取决于模板和实现，但训练与推理必须一致；`<eos>` 通常应计入，否则模型可能学不会适时停止。

另一种做法是对完整序列计算损失。它实现简单，也能训练模型复现用户侧文本，但会把容量和梯度花在“预测用户说什么”上。在纯指令跟随场景中，通常优先使用 response-only loss。

> 对 causal language model，代码中通常会将 logits 与 labels 错开一位：位置 $t-1$ 的输出用于预测位置 $t$ 的 token。很多框架已在模型内部完成 shift，手动再做一次会产生错位错误。

### 3.5 Teacher forcing

训练第 $i$ 条样本的第 $t$ 个回答 token 时，模型看到的是参考答案的真实前缀 $y_{i,\lt t}$，而不是自己刚生成的前缀，这称为 **teacher forcing**。它能并行、稳定地训练，却也带来训练与生成之间的差异：推理时模型一旦生成错误，后续 token 将建立在错误前缀上。高质量、多样化的数据和生成式评测因此十分重要，不能只看训练 loss。

## 4. 一次训练迭代发生了什么

```text
原始样本
  ↓ 清洗、去重、格式化
messages / instruction-response
  ↓ chat template + tokenizer
input_ids、attention_mask、labels（含 loss mask）
  ↓ 模型前向计算
每个位置的 next-token 概率
  ↓ 交叉熵损失
反向传播 → 参数更新
```

框架无关的伪代码如下：

```python
for batch in dataloader:
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],  # 不参与损失的位置通常填为 -100
    )
    loss = outputs.loss / gradient_accumulation_steps
    loss.backward()

    if should_update:
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
```

实践中还常用混合精度、梯度累积、gradient checkpointing 和 sequence packing 来降低显存或提高吞吐。Packing 是把多条短样本装入同一固定长度序列；使用时需要确保样本边界、attention mask、position ids 和 loss mask 都符合所用模型与框架的约定，避免不同样本意外互相“看见”。

## 5. 全参数微调、LoRA 与 QLoRA

SFT 描述的是**训练数据和目标函数**，全参数微调、LoRA、QLoRA 描述的是**哪些参数被更新以及如何节省资源**，两者不是同一层面的概念。

| 方法 | 更新内容 | 主要优点 | 主要代价或限制 |
| --- | --- | --- | --- |
| 全参数微调 | 所有模型参数 | 调整空间最大 | 显存、存储和训练成本高 |
| LoRA | 冻结原权重，训练低秩增量矩阵 | 参数少，便于保存和切换适配器 | rank、目标层等配置会影响效果 |
| QLoRA | 量化冻结的基座权重，再训练 LoRA | 进一步降低显存 | 实现更复杂，速度未必更快 |

LoRA 将某个冻结权重矩阵 $W_0$ 的更新写成低秩形式 [7]：

$$
W=W_0+\Delta W,
\qquad
\Delta W=\frac{\alpha}{r}BA,
$$

设原权重矩阵 $W_0\in\mathbb{R}^{d_{\mathrm{out}}\times d_{\mathrm{in}}}$，其中 $\mathbb{R}^{a\times b}$ 表示由实数组成的 $a$ 行 $b$ 列矩阵。则：

- $W_0$ 是预训练得到并在 LoRA 训练中保持冻结的原始权重；
- $W$ 是加入 LoRA 增量后，模型前向计算实际使用的等效权重；
- $\Delta W$ 是 LoRA 学习到的权重增量，与 $W_0$ 具有相同形状；
- $d_{\mathrm{in}}$ 和 $d_{\mathrm{out}}$ 分别是该线性层的输入维度和输出维度；
- $A\in\mathbb{R}^{r\times d_{\mathrm{in}}}$ 和 $B\in\mathbb{R}^{d_{\mathrm{out}}\times r}$ 是可训练的低秩矩阵，因此 $BA$ 的形状与 $W_0$ 相同；
- $r$ 是低秩分解的秩，通常远小于 $d_{\mathrm{in}}$ 和 $d_{\mathrm{out}}$；
- $\alpha$ 是 LoRA scaling factor， $\alpha/r$ 用于缩放低秩更新的幅度。

这里采用常见的 $BA$ 记法；有些实现会交换 $A$、 $B$ 的命名，但只要矩阵形状和相乘顺序匹配，数学含义相同。LoRA 训练时通常只更新 $A$ 和 $B$，而 $W_0$ 保持不变。无论采用哪一种参数更新方式，前面的 SFT 交叉熵目标都可以保持不变。QLoRA 进一步以 4-bit 量化方式存储冻结的基座模型，并通过若干内存优化实现低成本微调 [8]。

## 6. 为什么数据质量通常比数据数量更重要

一条 SFT 样本不仅在传授答案，还在示范语气、格式、推理习惯和安全边界。错误答案、互相矛盾的规则、冗长套话也会被模型一并模仿。LIMA 等研究表明，在模型已经具备较强预训练能力时，数量有限但精心筛选的数据也能显著改善指令遵循能力 [6]；这不意味着“少量数据永远足够”，而是说明边际收益高度取决于质量与覆盖面。

构建数据集时至少应检查：

1. **正确性**：答案事实、计算过程和代码是否正确。
2. **相关性**：数据是否覆盖部署时真正关心的任务和用户表达。
3. **一致性**：相似输入的回答原则、拒答边界和格式是否一致。
4. **多样性**：任务、难度、语言、长度和表达方式是否过于单一。
5. **可学习性**：指令是否清楚，答案是否足以从输入与上下文推出。
6. **安全与合规**：是否包含隐私、凭据、未授权内容或高风险示范。
7. **数据隔离**：训练集是否与验证集、测试集或公开基准泄漏重合。

合成数据可以降低成本并扩大覆盖范围，但要警惕生成模型的固定措辞、系统性错误和“自我复制”。较稳妥的流程是：先定义任务与质量标准，再生成候选数据，经过规则过滤、去重、模型或人工审核，最后抽样复查。

## 7. 关键训练选择

### 7.1 截断与样本长度

超过最大序列长度时，不应不加区分地从尾部截断。若回答尾部和 `<eos>` 被截掉，模型既损失监督信号，也可能更难学会停止。可以按任务选择过滤超长样本、保留回答并截短上下文，或使用滑窗和更长上下文模型。

### 7.2 token 平均与样本平均

常见实现将一个 batch 中所有有效 token 的损失取平均，因此长回答天然贡献更多权重。若希望每条样本的重要性相近，可以先计算每条样本的平均 token loss，再对样本取平均。二者没有绝对优劣，但应根据任务目标明确选择，并在梯度累积或多卡训练时保持归一化正确。

### 7.3 学习率、训练轮数与遗忘

学习率过大或训练轮数过多，可能让模型过拟合固定模板、损害原有通用能力，甚至出现灾难性遗忘。通常应同时观察训练集和验证集曲线，定期在通用能力与目标任务上生成样例，而不是机械追求更低的训练损失。混入一部分高质量通用指令数据，有时能缓解领域微调造成的能力偏移。

## 8. 常见问题与排查方向

| 现象 | 常见原因 | 优先检查 |
| --- | --- | --- |
| 模型复述用户问题 | 用户 token 也参与了损失；数据中大量复述式答案 | labels 与 loss mask |
| 回答停不下来 | `<eos>` 缺失、被截断或未参与损失 | 模板、截断策略、labels |
| 推理效果远差于验证 loss | 训练和推理模板不同；只做了 teacher-forced 评测 | chat template 与生成式评测 |
| loss 异常为 0、NaN 或不下降 | labels 全为 `-100`、错位、学习率或数值精度问题 | 一个 batch 的 token、labels 和梯度 |
| 模型只会固定措辞 | 数据表达单一、重复过多或训练过久 | 去重、多样性和训练轮数 |
| 多轮角色混乱 | 角色 token 错误；监督了不该预测的消息 | 模板及逐 token mask 可视化 |
| 领域能力提升但通用能力下降 | 数据分布过窄或参数更新过强 | 混合数据、学习率、训练轮数 |

训练前最好随机解码若干条 `input_ids`，并逐 token 查看 labels：只有模型应该学习生成的位置才保留 token id，其余位置应被 ignore index（通常是 `-100`）屏蔽。这一步能提前发现大量“代码正常运行、实际训练目标错误”的问题。

## 9. 如何评估一个 SFT 模型

评估应至少包含三层：

1. **训练健康度**：训练/验证 loss、梯度范数、学习率、吞吐和显存是否正常。
2. **目标任务能力**：在隔离测试集上计算准确率、F1、ROUGE、代码通过率等与任务匹配的指标。
3. **真实生成质量**：使用与部署一致的模板和解码参数，检查正确性、指令遵循、格式、事实性、安全性与回答长度；必要时做人评或成对比较。

困惑度（perplexity）是平均 token 负对数似然的指数形式：

$$
\mathrm{PPL}=\exp(\mathcal{L})
$$

其中 $\mathrm{PPL}$ 表示困惑度， $\exp(\cdot)$ 表示以自然常数 $e$ 为底的指数函数， $\mathcal{L}$ 表示使用自然对数计算的平均 token 负对数似然。若令 $\mathcal{L}=\mathcal{L}_{\mathrm{SFT}}(\theta)$，则这里的 PPL 只统计 $m_{i,t}=1$ 的监督位置。它能衡量模型对参考文本的预测能力，但不能单独代表回答是否有帮助。不同 tokenizer、不同 loss mask 或不同数据集上的 PPL 也不能直接横向比较。

## 10. SFT 的能力边界

SFT 学习的是示范数据中的条件分布：给定某类输入，参考回答通常长什么样。它很擅长建立基本行为模式，却有几个天然限制：

- 每个输入通常只有一个或少量“标准答案”，无法充分表达多个都正确的回答之间的细微偏好。
- 交叉熵会同等模仿答案里的优点和缺点，本身不知道什么更受人类偏爱。
- 对难以提供标准示范、但结果容易判定的任务，纯 SFT 可能无法充分探索更好的策略。
- 高质量专家示范昂贵，且训练数据不可能覆盖真实用户的所有输入。

因此，实践中常先用 SFT 建立稳定的指令遵循和输出格式，再使用 DPO、RLHF 或带可验证奖励的强化学习继续优化。SFT 通常也是这些方法的重要起点：若初始模型连基本格式都不稳定，后续偏好或奖励信号会更难利用。

## 11. 本章小结

- SFT 的本质是在示范数据上继续做条件 next-token prediction。
- 核心目标是带 loss mask 的 token 级交叉熵；数据模板和监督范围决定模型究竟在学什么。
- LoRA、QLoRA 是资源高效的参数更新方式，并不改变 SFT 的基本目标。
- 高质量、覆盖真实场景且格式一致的数据，通常比盲目扩充低质量数据更重要。
- 评估不能只看 loss，必须使用部署模板进行真实生成，并检查目标能力与通用能力。

## 附录 A：为什么最大似然估计会得到 SFT 目标函数

本附录从统计建模的角度推导 SFT 目标函数。核心思路可以先概括为一句话：**既然训练数据给出了在输入 $x_i$ 下希望模型生成的回答 $y_i$，就调整模型参数，使这些已观察到的回答在模型中出现的概率尽可能大。**

### A.1 建模目标与基本假设

假设现实中存在一个未知的真实条件分布 $q(y\mid x)$。给定输入 $x$，它描述了高质量回答 $y$ 可能以多大概率出现。我们无法直接知道 $q$，只能观察从中采集或近似构造出的训练集

$$
\mathcal{D}=\{(x_i,y_i)\}_{i=1}^{N}.
$$

我们用参数为 $\theta$ 的模型分布 $p_\theta(y\mid x)$ 去近似 $q(y\mid x)$。推导采用以下假设：

1. 每个 $y_i$ 都是在给定 $x_i$ 后得到的一条理想回答示范。
2. 给定各自的输入后，不同训练样本相互独立。
3. 训练时把输入 $x_i$ 视为已知条件，只估计回答的条件分布，因此这里使用的是**条件最大似然估计**。

符号 $q$ 表示未知的数据分布， $p_\theta$ 表示我们能够训练的模型分布。这里需要区分两个层次的目标：

- **总体统计目标**：同一个输入 $x$ 往往可以对应多个合理回答。理论上，我们希望模型的条件分布 $p_\theta(y\mid x)$ 尽可能接近数据所代表的回答分布 $q(y\mid x)$，而不是把某一个固定句子当成唯一正确答案。
- **有限样本上的训练信号**：实际 SFT 数据通常只为每个输入提供一条或少量参考回答。因此，对单条样本 $(x_i,y_i)$，训练只能直接提高已观察回答 $y_i$ 及其各个 token 的条件概率，并没有看到该输入下所有合理回答的完整分布。

模型仍可能学会生成训练集中未出现过的合理表达，这是因为不同样本共享同一组模型参数，而且预训练已经赋予模型语言规律和表达能力。需要注意， $q(y\mid x)$ 更准确地说是**理想的数据生成分布**；有限训练集只是它的一组样本。若数据存在偏差、错误或覆盖不足，训练集所体现的经验分布也不会等同于客观世界中所有正确回答的分布。

### A.2 从单条样本到整个训练集的似然

对于一条样本 $(x_i,y_i)$，模型在给定 $x_i$ 时生成已观察回答 $y_i$ 的概率是

$$
p_\theta(y_i\mid x_i).
$$

这个值越大，说明当前模型越能解释“为什么在输入 $x_i$ 下观察到了回答 $y_i$”。在样本条件独立的假设下，模型同时生成训练集中全部目标回答的条件概率等于各样本概率的乘积：

$$
\mathscr{L}(\theta;\mathcal{D})
=\prod_{i=1}^{N}p_\theta(y_i\mid x_i).
$$

这里的 $\mathscr{L}(\theta;\mathcal{D})$ 称为**似然函数**。它和概率使用相同的数值表达式，但观察角度不同：

- 讨论概率时，参数 $\theta$ 固定，把数据看作可能变化的结果；
- 讨论似然时，已观察到的数据 $\mathcal{D}$ 固定，把参数 $\theta$ 看作需要选择的变量。

最大似然估计选择使训练数据似然最大的参数：

$$
\hat{\theta}_{\mathrm{MLE}}
=\underset{\theta}{\arg\max}
\mathscr{L}(\theta;\mathcal{D})
=\underset{\theta}{\arg\max}
\prod_{i=1}^{N}p_\theta(y_i\mid x_i).
$$

其中 $\hat{\theta}_{\mathrm{MLE}}$ 表示最大似然估计得到的参数。直观上，如果一组参数总是给训练答案很低的概率，那么这组参数很难解释已经观察到的数据；反之，能为训练答案分配较高概率的参数更符合数据。

### A.3 为什么要取对数

大量小于 1 的概率连续相乘容易产生数值下溢，而且乘积也不方便求导。由于自然对数函数 $\log(\cdot)$ 严格单调递增，最大化一个正数与最大化它的对数会得到同一个最优参数：

$$
\underset{\theta}{\arg\max}\,\mathscr{L}(\theta;\mathcal{D})
=\underset{\theta}{\arg\max}\,\log\mathscr{L}(\theta;\mathcal{D}).
$$

利用“乘积的对数等于对数之和”，可得条件对数似然：

$$
\log\mathscr{L}(\theta;\mathcal{D})
=\sum_{i=1}^{N}\log p_\theta(y_i\mid x_i).
$$

机器学习优化器通常执行最小化，因此在前面加负号，把最大化对数似然改写为最小化**负对数似然**（Negative Log-Likelihood，NLL）：

$$
\hat{\theta}_{\mathrm{MLE}}
=\underset{\theta}{\arg\min}
\left[-\sum_{i=1}^{N}\log p_\theta(y_i\mid x_i)\right].
$$

这里并没有改变优化目标，只是把便于统计描述的“最大化”转换成了训练代码更常用的“最小化”。

### A.4 使用链式法则展开到 token 级别

目标回答 $y_i=(y_{i,1},\ldots,y_{i,T_i})$ 是一个 token 序列。根据概率链式法则，

$$
p_\theta(y_i\mid x_i)
=\prod_{t=1}^{T_i}
p_\theta(y_{i,t}\mid x_i,y_{i,\lt t}).
$$

将它代入负对数似然，并再次利用乘积的对数等于对数之和：

$$
\begin{aligned}
-\sum_{i=1}^{N}\log p_\theta(y_i\mid x_i)
&=-\sum_{i=1}^{N}
\log\prod_{t=1}^{T_i}
p_\theta(y_{i,t}\mid x_i,y_{i,\lt t}) \\
&=-\sum_{i=1}^{N}\sum_{t=1}^{T_i}
\log p_\theta(y_{i,t}\mid x_i,y_{i,\lt t}).
\end{aligned}
$$

这一步说明，**最大化完整回答的条件似然，等价于最大化回答中每个真实 token 的条件对数概率之和。** 自回归语言模型的 next-token prediction 恰好能逐项计算这些条件概率，因此可以直接使用梯度下降进行训练。

不同样本的回答长度不同。为了让损失尺度不随有效 token 总数线性增长，通常除以回答 token 总数，得到平均 token 负对数似然：

$$
\mathcal{L}_{\mathrm{response}}(\theta)
=-
\frac{
\displaystyle\sum_{i=1}^{N}\sum_{t=1}^{T_i}
\log p_\theta(y_{i,t}\mid x_i,y_{i,\lt t})
}{
\displaystyle\sum_{i=1}^{N}T_i
}.
$$

除以一个与 $\theta$ 无关的正常数不会改变全数据目标的最优解，只会改变损失和梯度的整体尺度。这里采用 token 平均；若改为先对每条样本取平均再对样本取平均，长短样本的相对权重会发生变化，目标函数便不再完全相同。

### A.5 从回答序列推广到 loss mask

实际代码处理的是模板化后的完整序列 $z_i$，其中可能同时包含 system、user、assistant 和特殊 token。使用 $m_{i,t}$ 指定哪些位置需要监督，token 级负对数似然便写成

$$
\mathcal{L}_{\mathrm{SFT}}(\theta)
=-
\frac{
\displaystyle\sum_{i=1}^{N}\sum_{t=1}^{L_i}
m_{i,t}\log p_\theta(z_{i,t}\mid z_{i,\lt t})
}{
\displaystyle\sum_{i=1}^{N}\sum_{t=1}^{L_i}m_{i,t}
}.
$$

当 mask 只在完整的 assistant 回答及其结束 token 上取 1 时，上式就是这些回答在给定对话上下文下的平均条件负对数似然。若对全部非 padding token 令 $m_{i,t}=1$，则目标变为完整模板化序列的语言模型负对数似然。由此可见，loss mask 不只是实现细节，它决定了模型究竟对哪些 token 做最大似然学习。

### A.6 为什么负对数似然等于交叉熵

设词表为 $\mathcal{V}$。在位置 $(i,t)$，真实 token 是 $z_{i,t}$。把真实标签写成 one-hot 分布

$$
q_{i,t}(v)=
\begin{cases}
1, & v=z_{i,t},\\
0, & v\ne z_{i,t},
\end{cases}
\qquad v\in\mathcal{V}.
$$

其中 $v$ 表示词表中的任意候选 token， $q_{i,t}(v)$ 表示真实标签在候选 token $v$ 上的概率。真实分布 $q_{i,t}$ 与模型预测分布 $p_\theta(\cdot\mid z_{i,\lt t})$ 的交叉熵为

$$
\begin{aligned}
H(q_{i,t},p_\theta)
&=-\sum_{v\in\mathcal{V}}
q_{i,t}(v)\log p_\theta(v\mid z_{i,\lt t})\\
&=-\log p_\theta(z_{i,t}\mid z_{i,\lt t}).
\end{aligned}
$$

由于 one-hot 分布中只有真实 token 对应的项为 1，其余项全为 0，交叉熵正好退化为真实 token 的负对数概率。因此，在 SFT 的 next-token classification 中，**token 级交叉熵与 token 级负对数似然是同一个量。**

### A.7 从 KL 散度理解为什么 MLE 合理

还可以从“逼近真实分布”的角度理解最大似然。对真实条件分布 $q(y\mid x)$ 取期望，模型的期望负对数似然为

$$
\mathbb{E}_{q(x,y)}[-\log p_\theta(y\mid x)].
$$

其中 $q(x,y)=q(x)q(y\mid x)$ 表示输入与回答的真实联合分布。把 $q(y\mid x)$ 乘进再除出，可以得到

$$
\begin{aligned}
\mathbb{E}_{q(x,y)}[-\log p_\theta(y\mid x)]
&=\mathbb{E}_{q(x,y)}[-\log q(y\mid x)] \\
&\quad+\mathbb{E}_{q(x,y)}
\left[\log\frac{q(y\mid x)}{p_\theta(y\mid x)}\right] \\
&=H_q(Y\mid X)
+\mathbb{E}_{q(x)}
\left[D_{\mathrm{KL}}\bigl(q(\cdot\mid x)\,\|\,p_\theta(\cdot\mid x)\bigr)\right].
\end{aligned}
$$

这里：

- $\mathbb{E}$ 表示对指定分布求期望，也就是按该分布对可能结果取加权平均；
- $H_q(Y\mid X)$ 是真实分布的条件熵，只由数据分布 $q$ 决定，与模型参数 $\theta$ 无关；
- $D_{\mathrm{KL}}(q\|p_\theta)$ 是从真实分布 $q$ 到模型分布 $p_\theta$ 的 KL 散度，用于衡量两个分布的差异，并且始终大于或等于 0。

因为第一项不随 $\theta$ 变化，所以最小化期望负对数似然，等价于最小化真实回答分布与模型回答分布之间的条件 KL 散度。有限训练集上的平均负对数似然，可以看作这个期望目标的经验估计。这给出了使用 MLE 的更深层理由：**它不只是让模型记住某几条答案，而是在数据足够且模型合适时，推动整个模型条件分布逼近数据所代表的条件分布。**

### A.8 推导成立不等于数据目标一定正确

上述推导说明了“给定示范数据后，MLE 如何得到 SFT 损失”，但它不能保证示范数据本身正确、全面或符合真实用户偏好。如果训练集中只有一个参考答案，MLE 会提高这个答案及相似表达的概率，却无法知道其他合理答案是否更好；若答案包含错误，MLE 也会忠实地学习错误。因此，SFT 的统计目标是否有用，最终仍取决于数据质量、覆盖范围、loss mask 和评估方式。

## 参考文献

1. Wei, J., et al. [Finetuned Language Models Are Zero-Shot Learners](https://arxiv.org/abs/2109.01652). ICLR, 2022.
2. Sanh, V., et al. [Multitask Prompted Training Enables Zero-Shot Task Generalization](https://arxiv.org/abs/2110.08207). ICLR, 2022.
3. Ouyang, L., et al. [Training Language Models to Follow Instructions with Human Feedback](https://arxiv.org/abs/2203.02155). NeurIPS, 2022.
4. Chung, H. W., et al. [Scaling Instruction-Finetuned Language Models](https://arxiv.org/abs/2210.11416). Journal of Machine Learning Research, 2024.
5. Longpre, S., et al. [The Flan Collection: Designing Data and Methods for Effective Instruction Tuning](https://arxiv.org/abs/2301.13688). ICML, 2023.
6. Zhou, C., et al. [LIMA: Less Is More for Alignment](https://arxiv.org/abs/2305.11206). NeurIPS, 2023.
7. Hu, E. J., et al. [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685). ICLR, 2022.
8. Dettmers, T., et al. [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314). NeurIPS, 2023.
9. Taori, R., et al. [Stanford Alpaca: An Instruction-following LLaMA Model](https://crfm.stanford.edu/2023/03/13/alpaca.html). Stanford Center for Research on Foundation Models, 2023.
10. Goodfellow, I., Bengio, Y., and Courville, A. [Deep Learning, Chapter 5: Machine Learning Basics](https://www.deeplearningbook.org/contents/ml.html). MIT Press, 2016.
11. Murphy, K. P. [Probabilistic Machine Learning: An Introduction](https://probml.github.io/pml-book/book1.html). MIT Press, 2022.
