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

#### 3.1 BERT（2018, Transformer Encoder Based）：

2018年，谷歌推出了BERT（Bidirectional Encoder Representations from Transformers），采用了双向训练方法。

![image](https://github.com/user-attachments/assets/a126ee8a-84db-434a-936d-bf4085771f98)

**BERT的关键创新：**

_1）掩码语言建模（Masked Language Modeling — MLM）：_ 考虑整个句子的上下文。

_2）下一句预测（Next Sentence Prediction — NSP）：_ 在理解句子之间关系的任务中表现出色，如问答和自然语言推理。

**BERT的「文本理解」优势：** 在文本分类、命名实体识别（NER）、情感分析等语言理解任务中表现出色。

#### 3.2 GPT：生成式预训练和自回归文本生成(2018-2020, Transformer Decoder Based）：

通过自回归预训练实现生成能力。

![image](https://github.com/user-attachments/assets/3a81fcd8-9697-4768-ad55-afe7d9bbe5ad)

**a. GPT (2018)：**

GPT的关键创新：

_1）单向自回归训练：_ 模型仅基于前面的Token预测下一个Token。

_2）[下游任务的微调](https://zhuanlan.zhihu.com/p/707913005)：_ 在不需要特定任务架构的情况下针对特定下游任务进行微调。只需添加一个分类头或修改输入格式。

GPT的优势：文本补全、摘要生成、对话生成、情感分析、机器翻译和问答等任务。

**b. GPT-2 (2019)：**

零样本(Zero-shot)能力：在没有任何特定任务微调的情况下执行任务。

![image](https://github.com/user-attachments/assets/cdf5f02f-4b65-4d32-8968-3098f8e6dd5b)

GPT-2优势：生成连贯的文章、回答问题，甚至在语言之间翻译文本，尽管没有明确针对这些任务进行训练。

**c. GPT-3 (2020)：**

超大规模：1750亿参数(175B parameters)。

少样本(Few-short)和零样本(Zero-short)学习能力：在推理时只需提供最少或无需示例即可执行任务。

GPT-3优势：创意写作、编程和复杂推理任务。

![image](https://github.com/user-attachments/assets/89a663f0-9fc4-43a3-8525-f883b91b3883)

![image](https://github.com/user-attachments/assets/6de8c6a1-9b3e-403d-adb2-73b1a38b2a93)

**d. Scaling Law：** 模型性能随着模型参数量、数据量和用于训练的计算量的指数级增加而平稳提高。

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


### 5. 多模态模型：连接文本、图像及其他 (2023–2024)

![image](https://github.com/user-attachments/assets/a06fd785-32e7-4978-bd20-793501fcef06)


#### 5.1 GPT-4V：视觉遇见语言

2023年，OpenAI推出了GPT-4V，将GPT-4的语言能力与先进的计算机视觉相结合。

#### 5.2 GPT-4o：全模态前沿

2024年初，GPT-4o通过整合音频和视频输入进一步推进了多模态。

### 6. 开源和开放权重模型 (2023–2024)

**开放权重LLMs：** 适合快速部署，如Meta AI的LLaMA系列。

**开源模型使底层代码和结构公开可用：** 这允许全面理解、修改和定制模型，促进创新和适应性，如OPT和BERT。

**社区驱动的创新：** 像Hugging Face这样的平台促进了协作，LoRA等工具使高效的微调成为可能。

![image](https://github.com/user-attachments/assets/fa6b6a41-8134-48a0-acb6-d663b11114c1)


### 7. 推理模型 (2024)

![image](https://github.com/user-attachments/assets/b6c2a6f0-4841-4c48-b066-5f2b27af3610)

#### 7.1 OpenAI-o1：推理能力飞跃(2024)

OpenAI-o1（2024年12月5日）与OpenAI-o3（2025年1月31日）在复杂数学和编程任务表现出色。

![image](https://github.com/user-attachments/assets/704807f8-f5ac-40ae-abbb-594c2a225787)

**长链思维（Long CoT） ：** 使模型能够将复杂问题分解为更小的部分，批判性地评估其解决方案。

但这些CoTs对用户是隐藏的，用户看到的是一个总结性的输出。


### 8. 成本高效的推理模型：DeepSeek-R1 (2025)

GPT-4o和OpenAI-o1这样的最先进LLM模型的闭源性质限制了对尖端AI的「普及化」。

#### 8.1 DeepSeek-V3 (2024–12)

DeepSeek-V3与OpenAI的ChatGPT等顶级解决方案相媲美，但开发成本显著降低。

**DeepSeek-V3的关键创新：**

_1）多头潜在注意力（Multi-head Latent Attention — MLA）：_ 通过压缩KV来减少内存使用。

_2）DeepSeek专家混合（DeepSeekMoE）：_ 提高效率并平衡专家利用率。

_3）多Token预测 (Multi-Token Prediction — MTP)：_ 提高生成速度，增强模型生成连贯且上下文相关的输出。

![image](https://github.com/user-attachments/assets/6acd9cd7-2f30-4094-a82c-9ef6a3fca068)

DeepSeek-V3的价格为每百万输出标记2.19美元，约为OpenAI类似模型成本的1/30。

#### 8.2 DeepSeek-R1-Zero 和 DeepSeek-R1 (2025–01)

展示了卓越的推理能力，训练成本极低。

**a. DeepSeek-R1-Zero：**

组相对策略优化（Group Relative Policy Optimization — GRPO）：PPO=>GRPO

![image](https://github.com/user-attachments/assets/2afa2dd9-3f54-4398-8daa-e7df15d43de3)


**b. DeepSeek-R1：** 

利用生成的推理数据SFT，第二轮RL训练。

![image](https://github.com/user-attachments/assets/95979e65-1086-4052-839c-f18056bacd2c)

**推理模型与非推理模型（生成模型、指令模型）**

<img width="1219" alt="image" src="https://github.com/user-attachments/assets/7f59484f-2003-4eaf-9067-fa66137ade78" />

<img width="1230" alt="image" src="https://github.com/user-attachments/assets/bf86cd96-34e9-4c22-8e67-5d369ace0384" />

<img width="1218" alt="image" src="https://github.com/user-attachments/assets/9b9e06bf-4eca-4f8b-8f96-e240c995ced5" />

<img width="1214" alt="image" src="https://github.com/user-attachments/assets/7193afdd-4f30-41ca-97fa-2ef59a2c783b" />

提示语没有那么复杂，直接对推理模型讲需求即可。

<img width="1196" alt="image" src="https://github.com/user-attachments/assets/58e2c1a5-b0dd-4a51-8f12-334356638ac8" />

<img width="1030" alt="image" src="https://github.com/user-attachments/assets/54d3c7d7-0704-473c-8944-0d3ef159b2ac" />

提示语：

<img width="1248" alt="image" src="https://github.com/user-attachments/assets/fac46563-253f-4b79-b64f-b04388924fbd" />


**c. 蒸馏DeepSeek模型：**

参数范围从15亿到700亿。

![image](https://github.com/user-attachments/assets/91c989d5-432f-4ae0-9bfa-d1401377810f)


DeepSeek-R1在各种基准测试中表现出竞争力，包括数学、编码、常识和写作。

相比OpenAI的o1模型等竞争对手提供了显著的成本节省，使用成本便宜20到50倍。

![image](https://github.com/user-attachments/assets/83ad8642-51af-4661-b094-7a7dc8812e73)


DeepSeek-R1的引入，使先进LLMs得以「普及化」。

### 9. Timeline

模型枚举：https://docs.google.com/spreadsheets/d/1kc262HZSMAWI6FVsh0zJwbB-ooYvzhCHaHcNUiA0_hY/edit?gid=1158069878#gid=1158069878

时间线：https://lifearchitect.ai/timeline/

排名：https://lifearchitect.ai/models-table/


## 二、DeepSeek的原理与相关技术

[DeepSeek核心技术大揭秘](https://mp.weixin.qq.com/s/K6FiAXFF_mgoS9ZnMoQ78w)

[DeepSeek关键技术详解](https://mp.weixin.qq.com/s/9Hlc_cbN8T5MLb_Pdi9K-w)

[揭开DeepSeek-R1的神秘面纱：GRPO 核心技术详解](https://mp.weixin.qq.com/s/jlduUciOX00mTpcvy-rX2g)

![image](https://github.com/user-attachments/assets/c5503a8e-e312-43f9-944b-4ef8ea0b1359)

### 1. R1一下：利用推理数据，二次SFT与RL
![image](https://github.com/user-attachments/assets/d5110520-7525-4634-b6b3-55a28d118bb3)

![image](https://github.com/user-attachments/assets/20a48d13-eb08-4191-9c90-80f3f853747b)

### 2. GRPO

**a. 强化学习基础概念**

[王树森深度强化学习基础](https://www.bilibili.com/video/BV1YxsxeEEmc?vd_source=e784ab20b5ab1efbef95dd71553d2844)

智能体（Agent）能够在与环境（Environment）的交互中通过试错来学习最优策略。智能体在环境中执行行动（Action），并根据行动的结果接收反馈，即奖励（Reward）。这些奖励信号指导智能体调整其策略，以最大化长期累积奖励。

![image](https://github.com/user-attachments/assets/6d98789e-1ddf-465a-be03-29e34cf2dc76)

**b.PPO：**

Policy 模型（又称 Actor）：输入一段上文，输出下一个token的概率分布。该模型需要训练，是我们最终得到的模型。输出下一个token即为Policy模型的“行为”。

Value 模型（又称 Critic）：用于预估当前模型回复的总收益。该总收益不仅局限于当前token的质量，还需要衡量当前token对后续文本生成的影响。该模型需要训练。

Reward 模型：事先用偏好数据进行训练，用于对Policy模型的预测进行打分，评估模型对于当前输出的即时收益。

Reference 模型：与 Policy 模型相同，但在训练过程中不进行优化更新，用于维持模型在训练中的表现，防止在更新过程中出现过大偏差。

**c.GRPO：**

PPO 在计算成本和训练稳定性方面仍然存在一定的挑战。GRPO 算法对此进行了优化，其核心目标是去除 Value 模型，以此来减少训练的计算资源。

GRPO 的方法是通过，大模型根据当前的上文输入进行多次采样，生成多个预测结果，并分别使用 Reward 模型对这些预测结果进行评分得到，最后取这些评分的平均值来替代 Value 模型的预期总收益估计。通过这种方式，GRPO 在训练过程中可以减少一个模型的前向和反向传播计算，从而降低计算资源的消耗。

![image](https://github.com/user-attachments/assets/75e2ae83-c01b-4bfc-832d-1d7ad33903b3)

### 3. Long-CoT：慢思考

CoT指的是一种推理过程，其中模型在生成最终答案之前，先逐步推导出一系列的中间步骤或子目标。这些中间步骤构成了一个“思维链”，最终引导模型得到正确的结果。它模仿人类的推理过程，即人们往往在解决问题时不是直接得出答案，而是通过一系列的思考、分析和推理步骤。

Long-CoT（长思考/慢思考）是CoT的一种扩展形式。传统的CoT方法通过将复杂问题分解为一系列中间推理步骤来引导模型进行逐步推理。而Long-CoT则进一步扩展了这种思路，使得思考过程和输出的答案解耦，可以通过更长的上下文和更复杂的推理路径（在思考过程中通过加入问题复述、思考回顾、反思、知识回忆、公式化等思考节点）来增强模型的推理能力。

![image](https://github.com/user-attachments/assets/93eed452-c21c-4a07-a493-35215957eba3)

**Aha Moment（顿悟时刻）：**

![image](https://github.com/user-attachments/assets/4b03bf60-3bed-4ac2-a730-df44ee8c98fb)

### 4. DeepSeek MoE 架构

![image](https://github.com/user-attachments/assets/9a17f19d-4ecd-484d-a06a-753d64c9bb3c)

**无辅助损失的负载均衡：**

<img width="443" alt="image" src="https://github.com/user-attachments/assets/1c4e5b04-5de6-41e7-9710-3f15b869243c" />

### 5. Multi-Token Prediction，MTP

当前主流的采用自回归的大模型都是单 token 预测。即根据当前上文预测下一个最可能的 token。而 MTP 的核心思想是让模型一次性预测多个 token，以提升了模型的训练效率、生成质量和推理速度。

![image](https://github.com/user-attachments/assets/97197ebb-8b3f-4257-85d8-4b9f61976e1d)


### 6. 多头隐式注意力（Multi-Head Latent Attention，MLA）

低秩分解

![image](https://github.com/user-attachments/assets/7986cad0-853b-4fcf-ade8-99f677603c00)

### 7. 原生稀疏注意力（Natively Sparse Attention，NSA）

![image](https://github.com/user-attachments/assets/263e30ef-18c0-4b54-a8de-c3851ffaee2c)

压缩路径（Compression）：将长序列划分为块级（如每块512个token），通过可学习的MLP压缩为粗粒度表示，捕获全局语义（类似“一目十行”）。

选择路径（Selection）：基于压缩块的注意力分数筛选出Top-N关键块，还原细粒度信息（类似“重点精读”）。

滑动窗口（Sliding Window）：固定窗口覆盖最近的局部token，确保上下文连贯性（类似“实时关注最新动态”）。

三条路径的结果通过门控机制动态加权融合，避免模型因局部信息优势而忽略全局

## 三、大模型应用及其发展趋势

1. Cursor

演示：https://www.cursor.com/cn/features

[AI编程神器Cursor，保姆级教程来了！](https://mp.weixin.qq.com/s/cDjJH1pa2rN7ENm7VxsSbg)

2. Trae

演示：https://www.trae.com.cn/
   
3. Manus与OWL（CAMEL-AI）

Manus：https://manus.im/

OWL：https://github.com/camel-ai/owl

OWL演示：https://mp.weixin.qq.com/s/0AWaSNynyjjY5TpdtKN-3w?forceh5=1
 
4. 两会政府工作报告：
   
1）持续推进“人工智能+”行动，将数字技术与制造优势、市场优势更好结合起来，支持**大模型广泛应用**，大力发展智能网联新能源汽车、人工智能手机和电脑、智能机器人等新一代**智能终端以及智能制造装备**。

2）培育壮大新兴产业、未来产业。深入推进战略性新兴产业融合集群发展。建立未来产业投入增长机制，培育生物制造、量子科技、**具身智能**、6G等未来产业。


周鸿祎：
![image](https://github.com/user-attachments/assets/99c8fe15-3b1d-4d58-a4cc-6ee6aa63634f)

![image](https://github.com/user-attachments/assets/d6611850-cbbb-4425-b19f-49699c145848)

## 参考文献

1. 「大模型简史」：从Transformer（2017）到DeepSeek-R1（2025）：https://mp.weixin.qq.com/s/dSn-o0bsbi1c92915ObIaA

   英文原版：https://medium.com/@lmpo/a-brief-history-of-lmms-from-transformers-2017-to-deepseek-r1-2025-dae75dd3f59a
   
2. DeepSeek学习资料：https://zhi35com.feishu.cn/wiki/V90xwCFDsid6Ylknv9WcHYvlnig
3. 清华刘知远、复旦邱锡鹏：https://www.bilibili.com/video/BV1joPDerE4h/?vd_source=e784ab20b5ab1efbef95dd71553d2844
4. 浙江大学DeepSeek系列公开课：https://www.bilibili.com/video/BV1TUA8eYEaF?vd_source=e784ab20b5ab1efbef95dd71553d2844
