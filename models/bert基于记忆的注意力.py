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

#memory    
class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(config.hidden_size * 2, config.num_classes)  # 维度加倍
        
        # 记忆模块参数
        self.mem_slots = 20  # 记忆槽数量
        self.mem_size = config.hidden_size  # 记忆维度与BERT一致
        
        # 可训练记忆矩阵 [mem_slots, mem_size]
        self.memory = nn.Parameter(torch.randn(self.mem_slots, self.mem_size))
        
        # 注意力投影层
        self.query_proj = nn.Linear(config.hidden_size, self.mem_size)
        self.output_proj = nn.Linear(self.mem_size, config.hidden_size)
        
        # 缓存中间结果
        self.mem_attention = None
        self.hidden_states = None

    def forward(self, x):
        context = x[0]
        mask = x[2]
        
        # 获取BERT输出
        last_hidden_state, pooled = self.bert(context, 
                                            attention_mask=mask,
                                            output_all_encoded_layers=False)
        self.hidden_states = last_hidden_state

        # 记忆注意力计算
        batch_size = context.size(0)
        
        # 生成查询向量（使用pooled输出）
        query = self.query_proj(pooled)  # [B, D]
        
        # 计算记忆相关性得分
        scores = torch.matmul(query.unsqueeze(1), 
                            self.memory.t().unsqueeze(0).expand(batch_size, -1, -1))  # [B, 1, M]
        scores = scores.squeeze(1)  # [B, M]
        
        # 计算注意力权重
        mem_weights = F.softmax(scores, dim=1)
        self.mem_attention = mem_weights  # 保存记忆注意力
        
        # 生成记忆上下文
        mem_context = torch.matmul(mem_weights.unsqueeze(1), 
                                self.memory.unsqueeze(0).expand(batch_size, -1, -1))  # [B, 1, D]
        mem_context = self.output_proj(mem_context.squeeze(1))  # [B, D]
        
        # 与原始特征拼接
        combined = torch.cat([pooled, mem_context], dim=1)  # [B, 2D]
        
        # 分类输出
        logits = self.fc(combined)
        self.probabilities = F.softmax(logits, dim=1)
        return logits