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

#time    
import torch
import torch.nn as nn
from pytorch_pretrained import BertModel, BertTokenizer
import torch.nn.functional as F

class TemporalAttentionModel(nn.Module):
    def __init__(self, config):
        super(TemporalAttentionModel, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        
        # 时间注意力参数
        self.temporal_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.position_embedding = nn.Embedding(100, config.hidden_size)  # 最大时间步长100
        self.temporal_attn = nn.MultiheadAttention(config.hidden_size, num_heads=8)
        
        # 分类层
        self.fc = nn.Linear(config.hidden_size, config.num_classes)
        
        # 中间结果缓存
        self.temporal_attention_weights = None
        self.probabilities = None

    def forward(self, x):
        context = x[0]  # 输入token ids [batch, seq_len]
        mask = x[2]     # 注意力mask [batch, seq_len]
        
        # 获取BERT输出
        last_hidden, pooled = self.bert(context, attention_mask=mask)
        
        # 添加时间位置编码
        batch_size, seq_len = context.size()
        positions = torch.arange(seq_len, device=context.device).expand(batch_size, seq_len)
        pos_emb = self.position_embedding(positions)
        temporal_features = last_hidden + pos_emb
        
        # 时间维度投影
        projected = self.temporal_proj(temporal_features)  # [batch, seq_len, hidden]
        
        # 时间注意力计算
        attn_output, attn_weights = self.temporal_attn(
            projected.transpose(0, 1),  # [seq_len, batch, hidden]
            projected.transpose(0, 1),
            projected.transpose(0, 1),
            key_padding_mask=~mask.bool()
        )
        attn_output = attn_output.transpose(0, 1)  # [batch, seq_len, hidden]
        
        # 时间特征聚合
        temporal_context = torch.mean(attn_output, dim=1)  # [batch, hidden]
        
        # 分类输出
        logits = self.fc(temporal_context)
        self.probabilities = F.softmax(logits, dim=1)
        self.temporal_attention_weights = attn_weights
        
        return logits