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

    
#random
class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(config.hidden_size, config.num_classes)
        # 中间结果缓存
        self.attention_weights = None
        self.hidden_states = None
        self.probabilities = None

    def forward(self, x):
        context = x[0]  
        mask = x[2]     
        
        # 获取BERT输出
        last_hidden_state, pooled = self.bert(context, 
                                            attention_mask=mask,
                                            output_all_encoded_layers=False)
        self.hidden_states = last_hidden_state  # 保存隐藏状态
        
        # 硬注意力机制
        query = pooled.unsqueeze(1) 
        scores = torch.bmm(query, last_hidden_state.transpose(1, 2)).squeeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)
        
        # 计算注意力概率
        attn_probs = F.softmax(scores, dim=1)
        
        # 采样得到硬注意力索引
        indices = torch.multinomial(attn_probs, 1)  # (batch_size, 1)
        
        # 生成硬注意力的one-hot向量
        hard_attn = torch.zeros_like(attn_probs).scatter_(1, indices, 1.0)
        
        # 使用straight-through estimator保持梯度传播
        hard_attn = (hard_attn - attn_probs).detach() + attn_probs
        
        self.attention_weights = hard_attn  # 保存注意力权重
        
        # 组合向量（只关注选中的位置）
        context_vector = torch.bmm(hard_attn.unsqueeze(1), last_hidden_state).squeeze(1)
        combined = pooled + context_vector
        
        # 分类输出
        logits = self.fc(combined)
        self.probabilities = F.softmax(logits, dim=1)  # 保存概率分布
        return logits