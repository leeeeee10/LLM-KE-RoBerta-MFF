import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer

class Config(object):
    """配置参数"""
    def __init__(self, dataset):
        self.dataset = dataset
        self.model_name = 'bert_multihead'
        self.train_path = dataset + '/data/train.txt'
        self.dev_path = dataset + '/data/dev.txt'
        self.test_path = dataset + '/data/test.txt'
        self.class_list = [x.strip() for x in open(dataset + '/data/class.txt').readlines()]
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.require_improvement = 1000
        self.num_classes = len(self.class_list)
        self.num_epochs = 8
        self.batch_size = 64
        self.pad_size = 64
        self.learning_rate = 3e-5
        self.bert_path = './bert_pretrain'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768
        self.hidden_dropout_prob = 0.5
        self.num_attention_heads = 4  # 多头注意力头数
        self.layer_norm_eps = 1e-7
        self.weight_decay = 0.1
        self.max_grad_norm = 1.0

class MultiHeadAttentionPooling(nn.Module):
    """纯多头注意力池化（不含多特征融合）"""
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.attentions = nn.ModuleList([
            nn.Linear(hidden_size, 1) for _ in range(num_heads)
        ])
        
    def forward(self, hidden_states, mask):
        # 各头独立计算注意力
        all_weights = []
        for attn in self.attentions:
            scores = attn(hidden_states).squeeze(-1)  # [batch, seq_len]
            scores = scores.masked_fill(mask == 0, -1e9)
            weights = F.softmax(scores, dim=-1)
            all_weights.append(weights.unsqueeze(1))  # [batch, 1, seq_len]
        
        # 合并多头结果
        weights = torch.cat(all_weights, dim=1)  # [batch, heads, seq_len]
        pooled = torch.einsum('bhs,bsd->bhd', weights, hidden_states)  # [batch, heads, hidden]
        
        # 返回平均池化结果和平均注意力权重（用于可视化）
        return pooled.mean(dim=1), weights.mean(dim=1)  # [batch, hidden], [batch, seq_len]

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        # BERT主干
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True

        # 注意力池化层
        self.attention_pool = MultiHeadAttentionPooling(
            config.hidden_size, 
            config.num_attention_heads
        )
        
        # 输出层
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_classes)

        # 中间结果缓存（用于分析和可视化）
        self.attention_weights = None    # 最终注意力权重 [batch, seq_len]
        self.last_hidden_state = None   # BERT最后层输出 [batch, seq_len, hidden]
        self.pooled_output = None       # 池化后向量 [batch, hidden]
        self.probabilities = None       # 预测概率 [batch, num_classes]

    def forward(self, x):
        input_ids, _, attention_mask = x  # 解包输入
        
        # BERT编码
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        last_hidden = outputs.last_hidden_state
        
        # 多头注意力池化
        pooled, attn_weights = self.attention_pool(last_hidden, attention_mask)
        
        # 缓存中间结果
        self.last_hidden_state = last_hidden
        self.attention_weights = attn_weights
        self.pooled_output = pooled
        
        # 分类预测
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        self.probabilities = F.softmax(logits, dim=-1)
        
        return logits