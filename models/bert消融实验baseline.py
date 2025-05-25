import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer

class Config(object):
    """配置参数"""
    def __init__(self, dataset):
        self.dataset = dataset
        self.model_name = 'bert_baseline'
        self.train_path = dataset + '/data/train.txt'
        self.dev_path = dataset + '/data/dev.txt'
        self.test_path = dataset + '/data/test.txt'
        self.class_list = [x.strip() for x in open(dataset + '/data/class.txt').readlines()]
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_classes = len(self.class_list)
        self.num_epochs = 8
        self.batch_size = 64
        self.pad_size = 64
        self.learning_rate = 3e-5
        self.bert_path = './bert_pretrain'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768
        self.hidden_dropout_prob = 0.1

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, x):
        input_ids, attention_mask = x[0], x[2]
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits