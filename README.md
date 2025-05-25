# LLM-KE-RoBerta-MFF

## 项目概述✨
本项目 `LLM-KE-RoBerta-MFF` 专注于自然语言处理和大语言模型（LLM）相关的研究与开发。它涵盖了模型微调、知识增强检索以及文本分类等多个重要任务。通过整合不同的技术和模型架构，旨在提升自然语言处理任务的性能和效果，为相关领域的研究和应用提供有力支持。

## 项目结构📁
```
LLM-KE-RoBerta-MFF/
├── lora_rag_LLM/
│   ├── main.py               # 模型微调主程序，负责加载模型、准备数据集、进行模型微调以及保存微调后的模型
│   ├── rag.py                # 知识增强检索模块，实现从知识库中检索相关知识并融入原始提示
│   ├── DeepSeek-R1-Distill-Qwen-1.5B/  # 模型文件目录，存放用于微调的基础模型文件
│   ├── datasets.jsonl        # 数据集文件，包含用于模型微调的样本数据
│   ├── final_models/         # 最终保存的全量模型目录，保存微调后的完整模型
│   └── saved_models/         # 保存的LoRA模型目录，保存LoRA微调后的模型参数
├── CSTA-Corpus/
│   ├── keyword.txt           # 关键词文本文件，可能包含与文本分类任务相关的关键词信息
│   └── data/
│       └── class.txt         # 类别文本文件，记录文本分类任务中的类别名单
├── models/
│   ├── bert.py               # 基于Roberta的文本分类模型实现，采用多头注意力池化机制
│   ├── bert自适应.py          # 自适应Roberta文本分类模型实现，引入自适应注意力机制
│   ├── bert_RCNN.py          # BERT与RCNN结合的文本分类模型实现（代码未给出，但从结构推测）
│   ├── bert时间注意力.py       # 带时间注意力的BERT文本分类模型实现，添加时间注意力机制
│   └── bert_RNN.py           # BERT与RNN结合的文本分类模型实现（代码未给出，但从结构推测）
├── run.py                    # 运行脚本，负责加载数据集、初始化模型、进行训练和评估
├── train_eval.py             # 训练和评估脚本，包含模型训练和评估的具体逻辑
└── utils.py                  # 工具函数脚本，提供如构建数据集、迭代器等工具函数
```

## 主要功能模块

### 1. LoRA微调与知识增强检索（`lora_rag_LLM`）🧠

#### `main.py`
此文件是模型微调的核心程序，其主要功能如下：
- **模型加载**：使用 `transformers` 库的 `AutoModelForCausalLM` 从指定路径加载 `DeepSeek-R1-Distill-Qwen-1.5B` 模型，并支持模型量化（使用 `BitsAndBytesConfig`）以减少内存使用。
- **数据集准备**：从 `data_pre.py` 中获取样本数据，将其保存为 `datasets.jsonl` 文件，并使用 `datasets` 库加载和处理数据集，划分为训练集和验证集。
- **模型微调**：使用 `peft` 库的 `LoraConfig` 对模型进行LoRA微调，通过 `transformers` 库的 `Trainer` 进行训练，并设置训练参数，如训练轮数、批次大小等。
- **模型保存**：将微调后的LoRA模型保存到 `saved_models` 目录，将全量模型保存到 `final_models` 目录。

#### `rag.py`
该文件实现了知识增强检索（RAG）功能，主要步骤如下：
- **知识加载**：从指定文件夹中加载按类别组织的知识，提取提示和完成内容。
- **TF-IDF预处理**：使用 `TfidfVectorizer` 对提示进行向量化处理，构建TF-IDF矩阵。
- **知识检索**：支持多类别查询的检索，优先基于领域名称进行检索，若未检测到领域则使用TF-IDF检索。
- **提示增强**：将检索到的知识以特定格式融入原始提示，增强模型的回答能力。

### 2. 文本分类（`models` 和 `run.py`）📚

#### `models` 目录
该目录包含多种基于BERT的文本分类模型实现，以 `bert.py` 为例，其主要特点如下：
- **模型结构**：使用 `BertModel` 作为基础模型，添加多头注意力池化层，将最后四层的隐藏状态拼接并进行注意力池化，再通过两层MLP进行分类。
- **配置参数**：包含训练集、验证集、测试集路径，类别名单，学习率，批次大小等配置信息。

#### `run.py`
这是文本分类任务的运行脚本，主要功能如下：
- **参数解析**：使用 `argparse` 解析命令行参数，指定使用的模型（如 `bert`、`ERNIE` 等）。
- **数据加载**：调用 `utils.py` 中的 `build_dataset` 和 `build_iterator` 函数加载和处理数据集。
- **模型初始化**：根据指定的模型名称导入相应的模型类，并初始化模型。
- **模型训练**：调用 `train_eval.py` 中的 `train` 函数进行模型训练和评估。

## 环境要求🖥️
- **Python 3.x**：项目代码基于Python 3开发，确保Python版本兼容。
- **PyTorch**：用于深度学习模型的构建和训练。
- **Transformers**：提供预训练模型和相关工具，方便模型加载和微调。
- **Datasets**：用于数据集的加载和处理。
- **Scikit-learn**：提供机器学习工具，如TF-IDF向量器。
- **Numpy**：用于数值计算。

## 安装依赖💻
```bash
pip install torch transformers datasets scikit-learn numpy
```

## 使用方法

### 模型微调
1. 确保 `lora_rag_LLM/DeepSeek-R1-Distill-Qwen-1.5B` 目录下包含所需的模型文件。
2. 运行以下命令进行模型微调：
```bash
python lora_rag_LLM/main.py
```
微调后的LoRA模型将保存到 `lora_rag_LLM/saved_models` 目录，全量模型将保存到 `lora_rag_LLM/final_models` 目录。

### 知识增强检索
可以在代码中调用 `rag.py` 中的 `KnowMoveRAG` 类进行知识检索和提示增强：
```python
from lora_rag_LLM.rag import KnowMoveRAG

rag = KnowMoveRAG()
original_prompt = "生成智能制造与装备领域的关键词"
augmented_prompt = rag.augment_prompt(original_prompt)
print(augmented_prompt)
```

### 文本分类
运行以下命令进行文本分类任务：
```bash
python run.py --model bert
```
其中 `--model` 参数可以选择不同的模型，如 `bert`、`ERNIE` 等。

## 注意事项⚠️
- 在运行模型微调之前，请确保 `lora_rag_LLM/datasets.jsonl` 文件存在，或者可以根据需要修改 `main.py` 中的数据集加载部分。
- 文本分类任务的数据集路径和模型保存路径可以在 `models` 目录下的配置文件中进行修改。

## 贡献🤝
欢迎对本项目进行贡献，包括但不限于提出问题、提交代码、改进文档等。请遵循项目的贡献指南进行操作。

## 联系方式📧
如果您有任何问题或建议，请通过GitHub Issues与我们联系。
