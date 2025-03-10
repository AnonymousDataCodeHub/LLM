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

**预训练模型的兴起：** BERT和GPT两大家族展示了大规模预训练和微调范式的强大功能。

2018年，谷歌推出了BERT（Bidirectional Encoder Representations from Transformers），这是一种使用Transformer编码器(Encoder)的突破性模型，在广泛的NLP任务中取得了最先进的性能。













