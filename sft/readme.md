# SFT（Supervised Fine-Tuning）基础理论

监督微调（Supervised Fine-Tuning，SFT）通常是大模型完成预训练后，走向“会听指令、会按格式回答”的第一步。本章先解释 SFT 在做什么、为什么有效，以及实践中最容易出错的地方；暂不绑定某个训练框架。

## 1. SFT 在后训练流程中的位置

一个常见的大模型训练流程可以简化为：

```text
海量无标注文本 ──预训练──> 基座模型 ──SFT──> 指令模型 ──偏好优化──> 更符合人类偏好的模型
                                           （DPO、PPO、GRPO 等）
```

- **预训练（Pre-training）**：从海量文本中学习语言规律、知识和一定的推理能力，核心任务通常是“预测下一个 token”。
- **SFT**：使用人工编写、模型合成或业务沉淀的“输入—理想输出”样本，教模型按照指令完成任务。
- **偏好优化（Preference Optimization）**：进一步利用回答之间的偏好、奖励信号或可验证结果，让模型更有用、更安全或更符合特定目标。

三者并非完全割裂。SFT 仍然使用 next-token prediction，只是训练数据从通用文本变成了高质量示范。可以把它理解为：**预训练主要赋予能力，SFT 主要教模型在什么场景下、以什么方式调用这些能力。** Instruction tuning 通常就是以自然语言指令为数据形式的 SFT；聊天模型训练则是它的多轮对话版本。

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

### 2.1 单轮、多轮与工具调用

- **单轮数据**适合问答、分类、抽取、翻译和固定格式生成等任务。
- **多轮数据**还要教会模型利用上下文、处理指代，并保持角色边界。
- **工具调用数据**需要包含工具描述、参数格式、工具返回结果及最终回答，用示范教会模型“何时调用、如何调用”。

多轮对话中，可以只监督最后一个 assistant 回答，也可以监督所有 assistant 回答。后者能利用更多训练信号，但必须正确屏蔽 system、user、工具返回值以及轮次分隔符中不希望模型学习预测的部分。

## 3. SFT 的核心目标函数

设训练集为

$$
\mathcal{D}=\{(x_i,y_i)\}_{i=1}^{N},
$$

其中 $x_i$ 是指令或对话上下文，$y_i=(y_{i,1},\ldots,y_{i,T_i})$ 是目标回答。自回归语言模型把回答的条件概率分解为

$$
p_\theta(y_i\mid x_i)
=\prod_{t=1}^{T_i}p_\theta(y_{i,t}\mid x_i,y_{i,<t}),
$$

其中 $\theta$ 表示模型参数，$y_{i,<t}$ 表示目标回答中第 $t$ 个 token 之前的所有 token。

SFT 使用最大似然估计（Maximum Likelihood Estimation，MLE）：让参考答案在模型下出现的概率尽可能大。等价地，最小化 token 级负对数似然，也就是交叉熵损失：

$$
\mathcal{L}_{\mathrm{SFT}}(\theta)
=-\frac{1}{\sum_{i,t}m_{i,t}}
\sum_{i=1}^{N}\sum_{t=1}^{L_i}
m_{i,t}\log p_\theta(z_{i,t}\mid z_{i,<t}).
$$

这里：

- $z_i$ 是将输入和目标回答按模板拼接后的完整 token 序列；
- $m_{i,t}\in\{0,1\}$ 是 loss mask；只有 $m_{i,t}=1$ 的位置参与损失；
- $L_i$ 是模板化后序列的长度。

### 3.1 为什么需要 loss mask

以单轮样本为例：

| token 区域 | `<user>` 与用户问题 | `<assistant>` | 助手回答 | `<eos>` | padding |
| --- | --- | --- | --- | --- | --- |
| 常见 loss mask | 0 | 0 或 1 | 1 | 1 | 0 |

这种做法常称为 **completion-only loss** 或 **response-only loss**：输入负责提供条件，主要让助手回答承担损失。`<assistant>` 是否计入损失取决于模板和实现，但训练与推理必须一致；`<eos>` 通常应计入，否则模型可能学不会适时停止。

另一种做法是对完整序列计算损失。它实现简单，也能训练模型复现用户侧文本，但会把容量和梯度花在“预测用户说什么”上。在纯指令跟随场景中，通常优先使用 response-only loss。

> 对 causal language model，代码中通常会将 logits 与 labels 错开一位：位置 $t-1$ 的输出用于预测位置 $t$ 的 token。很多框架已在模型内部完成 shift，手动再做一次会产生错位错误。

### 3.2 Teacher forcing

训练第 $t$ 个 token 时，模型看到的是参考答案的真实前缀 $y_{<t}$，而不是自己刚生成的前缀，这称为 **teacher forcing**。它能并行、稳定地训练，却也带来训练与生成之间的差异：推理时模型一旦生成错误，后续 token 将建立在错误前缀上。高质量、多样化的数据和生成式评测因此十分重要，不能只看训练 loss。

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

其中秩 $r$ 通常远小于原矩阵维度，只训练 $A$ 和 $B$。无论采用哪一种参数更新方式，前面的 SFT 交叉熵目标都可以保持不变。QLoRA 进一步以 4-bit 量化方式存储冻结的基座模型，并通过若干内存优化实现低成本微调 [8]。

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
\mathrm{PPL}=\exp(\mathcal{L}).
$$

它能衡量模型对参考文本的预测能力，但不能单独代表回答是否有帮助。不同 tokenizer、不同 loss mask 或不同数据集上的 PPL 也不能直接横向比较。

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
