import torch
import torch.nn as nn
from pytorch_pretrained import BertModel, BertTokenizer
import torch.nn.functional as F
import numpy as np

class Config(object):
    """配置参数"""
    def __init__(self, dataset):
        self.dataset = dataset
        self.model_name = 'bert'
        self.train_path = dataset + '/data/train.txt'
        self.dev_path = dataset + '/data/dev.txt'
        self.test_path = dataset + '/data/test.txt'
        self.class_list = [x.strip() for x in open(
            dataset + '/data/class.txt').readlines()]
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.require_improvement = 1000
        self.num_classes = len(self.class_list)
        self.num_epochs = 3
        self.batch_size = 128
        self.pad_size = 32
        self.learning_rate = 5e-5
        self.bert_path = './bert_pretrain'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768
        self.max_seq_len = 512
        self.min_seq_len = 5
        self.do_balance = True
        self.max_check_samples = 50
        self.temperature = 0.07
        self.contrast_weight = 0.5

# class Model(nn.Module):
#     def __init__(self, config):
#         super(Model, self).__init__()
#         self.bert = BertModel.from_pretrained(config.bert_path)
#         for param in self.bert.parameters():
#             param.requires_grad = True
#         self.fc = nn.Linear(config.hidden_size, config.num_classes)
        
#         # 自注意力参数
#         self.query = nn.Linear(config.hidden_size, config.hidden_size)
#         self.key = nn.Linear(config.hidden_size, config.hidden_size)
#         self.value = nn.Linear(config.hidden_size, config.hidden_size)
        
#         # 中间结果缓存
#         self.attention_weights = None
#         self.hidden_states = None
#         self.probabilities = None

#     def forward(self, x):
#         context = x[0]  # [batch_size, seq_len]
#         mask = x[2]     # [batch_size, seq_len]
        
#         # BERT编码
#         last_hidden_state, pooled = self.bert(context, 
#                                             attention_mask=mask,
#                                             output_all_encoded_layers=False)
#         self.hidden_states = last_hidden_state

#         # 自注意力计算
#         Q = self.query(last_hidden_state)
#         K = self.key(last_hidden_state)
#         V = self.value(last_hidden_state)
        
#         # 注意力分数
#         scores = torch.bmm(Q, K.transpose(1, 2)) / (K.size(-1) ** 0.5)
        
#         # 应用mask
#         expanded_mask = mask.unsqueeze(1).expand(-1, scores.size(1), -1)
#         scores = scores.masked_fill(expanded_mask == 0, -1e9)
        
#         # 注意力权重
#         attn_weights = F.softmax(scores, dim=-1)
#         self.attention_weights = attn_weights
        
#         # 上下文向量
#         context_vector = torch.bmm(attn_weights, V)
        
#         # 残差连接
#         attended = context_vector + last_hidden_state
        
#         # 稳健池化
#         if attended.size(1) > 0:
#             pooled_output = attended[:, 0, :]  # CLS token
#         else:
#             pooled_output = attended.mean(dim=1)  # 降级方案
        
#         # 分类输出
#         logits = self.fc(pooled_output)
#         self.probabilities = F.softmax(logits, dim=1)
        
#         # 维度验证
#         assert logits.dim() == 2, f"输出维度错误: {logits.shape}"
#         assert logits.size(0) == context.size(0), "批量大小不一致"
        
#         return logits
# ------------ models/bert.py ------------
import torch
import torch.nn as nn
from pytorch_pretrained import BertModel

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.config = config
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(config.hidden_size, config.num_classes)
        
        # 自注意力参数
        self.query = nn.Linear(config.hidden_size, config.hidden_size)
        self.key = nn.Linear(config.hidden_size, config.hidden_size)
        self.value = nn.Linear(config.hidden_size, config.hidden_size)
        
        # 中间结果缓存
        self.attention_weights = None

    def forward(self, x):
        context = x[0]  # [batch_size, seq_len]
        mask = x[2]     # [batch_size, seq_len]
        
        # BERT编码
        last_hidden_state, _ = self.bert(context, attention_mask=mask)
        
        # 自注意力计算
        Q = self.query(last_hidden_state)
        K = self.key(last_hidden_state)
        V = self.value(last_hidden_state)
        
        # 注意力分数（加入温度系数）
        scores = torch.bmm(Q, K.transpose(1, 2)) / (self.config.temperature * (K.size(-1) ** 0.5))
        
        # 应用mask
        expanded_mask = mask.unsqueeze(1).expand(-1, scores.size(1), -1)
        scores = scores.masked_fill(expanded_mask == 0, -1e9)
        
        # 注意力权重
        attn_weights = F.softmax(scores, dim=-1)
        self.attention_weights = attn_weights
        
        # 上下文向量和残差连接
        context_vector = torch.bmm(attn_weights, V)
        attended = context_vector + last_hidden_state
        
        # 池化输出
        pooled_output = attended[:, 0, :]  # CLS token
        
        # 分类输出
        logits = self.fc(pooled_output)
        return logits