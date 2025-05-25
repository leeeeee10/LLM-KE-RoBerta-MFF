# coding: UTF-8
import torch
import torch.nn as nn
# from pytorch_pretrained_bert import BertModel, BertTokenizer
from pytorch_pretrained import BertModel, BertTokenizer
import torch.nn.functional as F  # 添加这一行


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
        self.quality_check_threshold = 0.6  # 可配置的置信度阈值
        self.max_check_samples = 50         # 每次最多检查样本数
        self.auto_correct_threshold = 0.65  # 新增自动校正置信度阈值
        self.enable_auto_correction = True  # 是否启用自动校正
        self.temperature = 0.07  # 对比学习温度系数
        self.contrast_weight = 0.5  # 对比损失权重
    
#soft
class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(config.hidden_size, config.num_classes)
        self.last_pooled = None  # 中间结果缓存

    def forward(self, x):
        context = x[0]  # 输入的句子token ids
        mask = x[2]     # 注意力掩码（batch_size, seq_len）
        
        # 获取BERT输出
        last_hidden_state, pooled = self.bert(context, 
                                            attention_mask=mask,
                                            output_all_encoded_layers=False)
        
        # 软注意力机制
        query = pooled.unsqueeze(1)  # [CLS]向量作为查询（batch_size, 1, hidden_size）
        
        # 计算注意力得分（点积）
        scores = torch.bmm(query, last_hidden_state.transpose(1, 2))  # (batch_size, 1, seq_len)
        scores = scores.squeeze(1)  # (batch_size, seq_len)
        
        # 处理padding的mask
        scores = scores.masked_fill(mask == 0, -1e9)  # 将padding位置的得分设为负无穷
        
        # 计算注意力权重
        attn_weights = F.softmax(scores, dim=1)  # (batch_size, seq_len)
        
        # 计算上下文向量（加权求和）
        context_vector = torch.bmm(attn_weights.unsqueeze(1), last_hidden_state).squeeze(1)
        
        # 组合原始[CLS]向量和注意力结果
        combined = pooled + context_vector  # 相加组合方式
        
        # 缓存中间结果（可选）
        self.last_pooled = combined
        
        # 最终分类
        out = self.fc(combined)
        return out


#初始
# class Model(nn.Module):

#     def __init__(self, config):
#         super(Model, self).__init__()
#         self.bert = BertModel.from_pretrained(config.bert_path)
#         for param in self.bert.parameters():
#             param.requires_grad = True
#         self.fc = nn.Linear(config.hidden_size, config.num_classes)
#         # 添加中间输出缓存
#         self.last_pooled = None  

#     def forward(self, x):
#         context = x[0]  # 输入的句子
#         mask = x[2]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
#         _, pooled = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)
#         self.last_pooled = pooled  # 缓存中间结果
#         out = self.fc(pooled)
#         return out

#LSTM+soft
# class Model(nn.Module):

#     def __init__(self, config):
#         super(Model, self).__init__()
#         self.bert = BertModel.from_pretrained(config.bert_path)
#         for param in self.bert.parameters():
#             param.requires_grad = True
#                 # 新增双向LSTM层
#         self.lstm = nn.LSTM(
#             input_size=config.hidden_size,
#             hidden_size=config.hidden_size // 2,  # 双向时每方向维度减半
#             bidirectional=True,
#             batch_first=True,
#             num_layers=1  # 可根据需要调整层数
#         )    
            
#         self.fc = nn.Linear(config.hidden_size, config.num_classes)
#         self.last_pooled = None  # 中间结果缓存

#     def forward(self, x):
#         context = x[0]  # 输入的句子token ids
#         mask = x[2]     # 注意力掩码（batch_size, seq_len）
        
#         # 获取BERT输出
#         bert_output, _ = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)
#         # 通过LSTM处理序列
#         lstm_output, (h_n, c_n) = self.lstm(bert_output)
#         # 生成新的聚合表示（双向LSTM的最终状态拼接）
#         if self.lstm.bidirectional:
#             # 前向和后向的最终隐藏状态拼接
#             h_forward = h_n[-2]  # 前向最后一个时间步
#             h_backward = h_n[-1] # 后向最后一个时间步
#             new_pooled = torch.cat([h_forward, h_backward], dim=1)
#         else:
#             new_pooled = h_n[-1]  # 单向LSTM取最后时间步

             
#         # 注意力机制
#         query = new_pooled.unsqueeze(1)  # [batch, 1, hidden_size]
#         scores = torch.bmm(query, lstm_output.transpose(1, 2)).squeeze(1)
#         scores = scores.masked_fill(mask == 0, -1e9)
#         attn_weights = F.softmax(scores, dim=1)
        
#         # 计算上下文向量
#         context_vector = torch.bmm(attn_weights.unsqueeze(1), lstm_output).squeeze(1)
        
#         # 组合表示
#         combined = new_pooled + context_vector
#         self.last_pooled = combined  # 缓存
        
#         # 分类输出
#         return self.fc(combined)