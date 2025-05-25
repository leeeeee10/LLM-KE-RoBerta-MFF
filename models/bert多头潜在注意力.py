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

    
#MLA
class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(config.hidden_size, config.num_classes)
        
        # 多头注意力参数
        self.num_heads = 8  # 头数
        self.head_dim = config.hidden_size // self.num_heads  # 每个头的维度
        
        # 定义QKV的线性投影层
        self.query = nn.Linear(config.hidden_size, config.hidden_size)
        self.key = nn.Linear(config.hidden_size, config.hidden_size)
        self.value = nn.Linear(config.hidden_size, config.hidden_size)
        
        # 最终输出投影层
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)
        
        # 中间结果缓存
        self.attention_weights = None
        self.hidden_states = None
        self.probabilities = None

    def forward(self, x):
        context = x[0]
        mask = x[2]
        
        # 获取BERT输出 [batch_size, seq_len, hidden_size]
        last_hidden_state, pooled = self.bert(context, 
                                            attention_mask=mask,
                                            output_all_encoded_layers=False)
        self.hidden_states = last_hidden_state

        # 生成QKV
        batch_size = context.size(0)
        q = self.query(pooled).view(batch_size, self.num_heads, self.head_dim)  # [B, h, d]
        k = self.key(last_hidden_state).view(batch_size, -1, self.num_heads, self.head_dim).permute(0,2,1,3)  # [B, h, L, d]
        v = self.value(last_hidden_state).view(batch_size, -1, self.num_heads, self.head_dim).permute(0,2,1,3)  # [B, h, L, d]

        # 计算注意力得分
        scores = torch.matmul(q.unsqueeze(2), k.transpose(-1, -2))  # [B, h, 1, L]
        scores = scores / (self.head_dim ** 0.5)
        
        # 处理mask
        mask = mask.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, L]
        scores = scores.masked_fill(mask == 0, -1e9)
        
        # 计算注意力权重
        attn_weights = F.softmax(scores, dim=-1)
        self.attention_weights = attn_weights  # 保存注意力权重 [B, h, 1, L]

        # 上下文向量计算
        context = torch.matmul(attn_weights, v)  # [B, h, 1, d]
        context = context.transpose(1,2).contiguous().view(batch_size, -1)  # [B, 1, h*d] -> [B, h*d]
        
        # 输出投影
        context_vector = self.out_proj(context)
        combined = pooled + context_vector

        # 分类输出
        logits = self.fc(combined)
        self.probabilities = F.softmax(logits, dim=1)
        return logits