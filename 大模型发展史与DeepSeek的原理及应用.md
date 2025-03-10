# 大模型发展史与DeepSeek的原理及应用

## 一、大模型发展史
![image](https://github.com/user-attachments/assets/3c588676-f2fc-41f7-a310-4f956cd161e1)

### 1. 大型语言模型（LLMs）
**定义：** LLMs 通常包含数十亿个参数（例如，GPT-3 拥有 1750 亿个参数）的语言模型（LMs），旨在处理、理解和生成类似人类的语言。它们从大型数据集中学习模式和结构，使得能够产生连贯且上下文相关的文本。

**自回归语言模型：** 大多数LLMs以「自回归方式」(Autoregressive)操作，这意味着它们根据前面的「文本」预测下一个「字」（或token／sub-word）的「概率分布」(propability distribution)。

**文本生成：** 大模型实际在做「文字接龙」。
![image](https://github.com/user-attachments/assets/c64bf432-4eb4-43e0-8f46-7bfc6f075fd5)

### 2. Transformer的提出 (2017)

**Transformer架构：** 2017年谷歌论文“Attention is All You Need”引入了Transformer架构，解决了早期模型如循环神经网络（RNNs）和长短期记忆网络（LSTMs）在长程依赖性和顺序处理方面的问题。

![image](https://github.com/user-attachments/assets/63d90d40-6b15-47aa-8efb-cbd3dcc88f53)

**Transformer架构的关键创新：** 

_1）自注意力机制 (Self-Attention)：_ 权衡每个Token相对于其他Token的重要性，使得模型能够动态关注输入的相关部分。

![image](https://github.com/user-attachments/assets/80566c40-c024-4158-b3e0-bfcf9991ed4b)

![image](https://github.com/user-attachments/assets/baf8fb18-ca8b-40fc-a54a-6915e5d8f7c2)

_2）多头注意力：_ 多个注意力头并行操作，每个头专注于输入的不同方面，实现更丰富的上下文表示。

![image](https://github.com/user-attachments/assets/db7c4c5a-7e39-4b79-9758-ad4dd65d0de7)

_3）前馈网络(FFN)和层归一化(Layer Norm)：_ 稳定了训练并支持更深的架构。


_4）位置编码：_ 保留顺序信息。

![image](https://github.com/user-attachments/assets/cd9f19e0-779d-4a2b-8371-301bedac2737)

Transformer架构的引入为构建大规模高效语言模型奠定了基础。


### 3. 预训练Transformer模型时代 (2018–2020)

BERT和GPT两大家族展示了大规模预训练和微调范式的强大功能。

#### 3.1 BERT（2018, Transformer Encoder Based）： 2018年，谷歌推出了BERT（Bidirectional Encoder Representations from Transformers），采用了双向训练方法。

![image](https://github.com/user-attachments/assets/a126ee8a-84db-434a-936d-bf4085771f98)

**BERT的关键创新：**

_1）掩码语言建模（Masked Language Modeling — MLM）：_ 考虑整个句子的上下文。

_2）下一句预测（Next Sentence Prediction — NSP）：_ 在理解句子之间关系的任务中表现出色，如问答和自然语言推理。

**BERT的「文本理解」优势：** 在文本分类、命名实体识别（NER）、情感分析等语言理解任务中表现出色。

#### 3.2 GPT：生成式预训练和自回归文本生成(2018-2020, Transformer Decoder Based）：** 通过自回归预训练实现生成能力。

![image](https://github.com/user-attachments/assets/3a81fcd8-9697-4768-ad55-afe7d9bbe5ad)

**GPT (2018)：**

GPT的关键创新：

_1）单向自回归训练：_ 模型仅基于前面的Token预测下一个Token。

_2）[下游任务的微调](https://zhuanlan.zhihu.com/p/707913005)：_ 在不需要特定任务架构的情况下针对特定下游任务进行微调。只需添加一个分类头或修改输入格式。

GPT的优势：文本补全、摘要生成、对话生成、情感分析、机器翻译和问答等任务。

**GPT-2 (2019)：**

零样本(Zero-shot)能力：在没有任何特定任务微调的情况下执行任务。

![image](https://github.com/user-attachments/assets/cdf5f02f-4b65-4d32-8968-3098f8e6dd5b)

GPT-2优势：生成连贯的文章、回答问题，甚至在语言之间翻译文本，尽管没有明确针对这些任务进行训练。

**GPT-3 (2020)：**

超大规模：1750亿参数(175B parameters)。

少样本(Few-short)和零样本(Zero-short)学习能力：在推理时只需提供最少或无需示例即可执行任务。

GPT-3优势：创意写作、编程和复杂推理任务。

![image](https://github.com/user-attachments/assets/89a663f0-9fc4-43a3-8525-f883b91b3883)

![image](https://github.com/user-attachments/assets/6de8c6a1-9b3e-403d-adb2-73b1a38b2a93)

Scaling Law：模型性能随着模型参数量、数据量和用于训练的计算量的指数级增加而平稳提高。

![image](https://github.com/user-attachments/assets/de73074b-2139-457c-975a-462dc6e24b30)


### 4. 后训练对齐：弥合AI与人类价值观之间的差距 (2021–2022)

AI幻觉（Hallucination）：LLM生成与事实不符、无意义或与输入提示矛盾的内容。

#### 4.1 监督微调 (SFT)

SFT：在高质量的输入-输出对上训练模型。

![image](https://github.com/user-attachments/assets/f1ddc5f0-c933-488a-8bbe-4ee54e8120ac)

确保模型学会生成准确且符合上下文的响应。

![image](https://github.com/user-attachments/assets/8e146d96-c276-4f8b-a9d9-6d501803e16e)

**SFT的局限性：**

_1）可扩展性：_ 收集这些高质量输入输出对是劳动密集型且耗时的。

_2）性能：_ 简单模仿人类行为并不能保证模型会超越人类表现或在未见过的任务上很好地泛化。

#### 4.2 基于人类反馈的强化学习 (RLHF)

**RLHF的两个关键阶段：**

_1）训练奖励模型：_ 人类对模型生成的多个输出进行排名，创建一个偏好数据集，用于训练奖励模型。

_2）使用强化学习微调LLM：_ 奖励模型使用近端策略优化（Proximal Policy Optimization - PPO）指导LLM的微调，学习生成更符合人类偏好和期望的输出。

这个两阶段过程 — — 结合SFT和RLHF — — 显著增强了模型生成可靠、符合人类输出的能力。

#### 4.3 ChatGPT：推进对话式AI (2022)

基于GPT-3.5和InstructGPT，OpenAI于2022年11月推出了ChatGPT。

![image](https://github.com/user-attachments/assets/4f86b3c9-4944-4972-9e4e-f847d8c7bfa5)

**ChatGPT的关键创新：**

_1）对话聚焦的微调：_ 在大量对话数据集上进行训练。

_2）RLHF：_ 生成不仅有用而且诚实和无害的响应。


「ChatGPT时刻」(ChatGPT moment)：ChatGPT的推出标志着AI的一个关键时刻。
















