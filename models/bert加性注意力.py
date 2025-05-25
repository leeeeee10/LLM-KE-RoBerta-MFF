import torch
import torch.nn as nn
# from pytorch_pretrained_bert import BertModel, BertTokenizer
from pytorch_pretrained import BertModel, BertTokenizer
import torch.nn.functional as F  # 添加这一行
import os

class Config(object):

    """配置参数"""
    def __init__(self, dataset):
        self.dataset = dataset
        self.model_name = 'bert'
        self.train_path = dataset + '/data/train.txt'                                # 训练集
        self.dev_path = dataset + '/data/dev.txt'                                    # 验证集
        self.test_path = dataset + '/data/test.txt'                                  # 测试集
        self.class_list = [x.strip() for x in open(
            dataset + '/data/class.txt').readlines()]                                # 类别名单
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'        # 模型训练结果
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备

        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)                         # 类别数
        self.num_epochs = 3                                             # epoch数
        self.batch_size = 128                                           # mini-batch大小
        self.pad_size = 32                                              # 每句话处理成的长度(短填长切)
        self.learning_rate = 5e-5                                       # 学习率
        self.bert_path = './bert_pretrain'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768
        self.do_balance = True  # 新增平衡开关
        self.max_check_samples = 50         # 每次最多检查样本数
        self.temperature = 0.07  # 对比学习温度系数
        self.contrast_weight = 0.5  # 对比损失权重

#Bahdanau     
class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        
        # 加性注意力参数
        self.attn_hidden = 512  # 注意力隐藏层维度
        self.attn = nn.Sequential(
            nn.Linear(config.hidden_size * 2, self.attn_hidden),
            nn.Tanh(),
            nn.Linear(self.attn_hidden, 1)
        )
        
        self.fc = nn.Linear(config.hidden_size, config.num_classes)
        
        # 中间结果缓存
        self.attention_weights = None
        self.hidden_states = None

    def forward(self, x):
        context = x[0]  # [batch_size, seq_len]
        mask = x[2]     # [batch_size, seq_len]
        
        # 获取BERT输出
        last_hidden_state, pooled = self.bert(context, 
                                            attention_mask=mask,
                                            output_all_encoded_layers=False)
        self.hidden_states = last_hidden_state  # [B, L, D]

        # 生成查询向量（扩展维度）
        query = pooled.unsqueeze(1).expand(-1, last_hidden_state.size(1), -1)  # [B, L, D]
        
        # 拼接查询和键
        energy = torch.cat([query, last_hidden_state], dim=2)  # [B, L, 2D]
        
        # 计算注意力能量
        attn_scores = self.attn(energy).squeeze(2)  # [B, L]
        
        # 掩码处理
        attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        # 归一化注意力权重
        attn_weights = F.softmax(attn_scores, dim=1)
        self.attention_weights = attn_weights
        
        # 生成上下文向量
        context_vector = torch.bmm(attn_weights.unsqueeze(1), last_hidden_state).squeeze(1)
        
        # 残差连接
        combined = pooled + context_vector
        
        # 分类输出
        logits = self.fc(combined)
        return logits