# LLM-KE-RoBerta-MFF

## 项目概述✨
本项目 `LLM-KE-RoBerta-MFF` 聚焦于自然语言处理与大语言模型（LLM）相关的研究和开发，涵盖了模型微调、知识增强检索、文本分类等多个方面的任务。通过整合不同的技术和模型，旨在提升自然语言处理任务的性能和效果。

## 项目结构📁
```
LLM-KE-RoBerta-MFF/
├── lora_rag_LLM/
│   ├── main.py               # 模型微调主程序
│   ├── rag.py                # 知识增强检索模块
│   ├── DeepSeek-R1-Distill-Qwen-1.5B/  # 模型文件目录
│   ├── datasets.jsonl        # 数据集文件
│   ├── final_models/         # 最终保存的全量模型
│   └── saved_models/         # 保存的LoRA模型
├── CSTA-Corpus/
│   ├── keyword.txt           # 关键词文本文件
│   └── data/
│       └── class.txt         # 类别文本文件
├── models/
│   ├── bert.py               # Roberta模型实现
│   ├── bert自适应.py          # 自适应Roberta模型实现
│   ├── bert_RCNN.py          # BERT与RCNN结合模型实现
│   ├── bert时间注意力.py       # 带时间注意力的BERT模型实现
│   └── bert_RNN.py           # BERT与RNN结合模型实现
├── run.py                    # 运行脚本
├── train_eval.py             # 训练和评估脚本
└── utils.py                  # 工具函数脚本
```

## 主要功能模块

### 1. LoRA微调与知识增强检索（`lora_rag_LLM`）🧠
- **`main.py`**: 负责加载模型、准备数据集、进行模型微调以及保存微调后的模型。使用了LoRA（Low-Rank Adaptation）技术对大语言模型进行高效微调，同时支持模型量化以减少内存使用。
- **`rag.py`**: 实现了知识增强检索（RAG）功能，通过TF-IDF向量空间模型和基于领域名称的检索方法，从知识库中检索相关知识，并将其融入到原始提示中，以增强模型的回答能力。

### 2. 文本分类（`models` 和 `run.py`）📚
- **`models` 目录**: 包含了多种基于BERT的文本分类模型实现，如BERT、BERT与RCNN结合、BERT与RNN结合等。这些模型通过不同的架构和注意力机制，提升文本分类的性能。
- **`run.py`**: 是文本分类任务的运行脚本，负责加载数据集、初始化模型、进行训练和评估。运行时可通过命令行参数指定使用的模型，如 `Bert` 或 `ERNIE`。

```python
# coding: UTF-8
import time
import torch
import numpy as np
from train_eval import train, init_network
from importlib import import_module
import argparse
from utils import build_dataset, build_iterator, get_time_dif

parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', type=str, required=True, help='choose a model: Bert, ERNIE')
args = parser.parse_args()

if __name__ == '__main__':
    dataset = 'CSTA-Corpus'  # 数据集

    model_name = args.model  # bert
    x = import_module('models.' + model_name)
    config = x.Config(dataset)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    start_time = time.time()
    print("Loading data...")
    train_data, dev_data, test_data = build_dataset(config)
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    test_iter = build_iterator(test_data, config, return_contents=True)  # 启用返回文本

    # train
    model = x.Model(config).to(config.device)
    train(config, model, train_iter, dev_iter, test_iter, train_data)
```

## 环境要求🖥️
- Python 3.x
- PyTorch
- Transformers
- Datasets
- Scikit-learn
- Numpy

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
