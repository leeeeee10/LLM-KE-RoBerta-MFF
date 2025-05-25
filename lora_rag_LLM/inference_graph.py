import os
import re
import heapq
from collections import defaultdict
from transformers import pipeline
import torch  # 添加缺失的PyTorch导入

class EnhancedKnowledgeGraph:
    def __init__(self, seed_file):
        self.nodes = {}
        self.global_vocab = defaultdict(int)
        self._load_seed_data(seed_file)
        self._init_priority_queue()

    def _load_seed_data(self, file_path):
        """加载种子词库数据"""
        with open(file_path, 'r', encoding='utf-8') as f:
            current_category = None
            for line in f:
                if line.startswith('#'):
                    current_category = line[1:].strip()
                    self.nodes[current_category] = {
                        'remaining': 30,
                        'keywords': set()
                    }
                elif current_category and line.strip():
                    keywords = [kw.strip() for kw in re.split(r'[，,、]', line)]
                    for kw in keywords:
                        if kw:
                            self.nodes[current_category]['keywords'].add(kw)
                            self.global_vocab[kw] += 1
                    # 更新剩余需求
                    current_count = len(self.nodes[current_category]['keywords'])
                    self.nodes[current_category]['remaining'] = max(0, 30 - current_count)

    def _init_priority_queue(self):
        """初始化优先队列"""
        self.heap = []
        for cat, data in self.nodes.items():
            if data['remaining'] > 0:
                heapq.heappush(self.heap, (-data['remaining'], cat))

    def get_next_category(self):
        """获取最需要补充的类别"""
        while self.heap:
            neg_remain, cat = heapq.heappop(self.heap)
            if self.nodes[cat]['remaining'] > 0:
                # 重新插入队列保持实时性
                heapq.heappush(self.heap, (neg_remain, cat))
                return cat
        return None

    def update_state(self, category, new_keywords):
        """更新图谱状态"""
        valid_kws = []
        node = self.nodes[category]
        
        for kw in new_keywords:
            kw = self._normalize_keyword(kw)
            # 双重过滤条件
            if (kw not in node['keywords'] and 
                self.global_vocab[kw] < 2 and 
                len(kw) >= 2 and len(kw) <= 12):
                valid_kws.append(kw)
                node['keywords'].add(kw)
                self.global_vocab[kw] += 1
                
        added = len(valid_kws)
        node['remaining'] = max(0, node['remaining'] - added)
        return valid_kws

    def _normalize_keyword(self, kw):
        """关键词规范化处理"""
        kw = re.sub(r'[\(（].*?[\)）]', '', kw)  # 移除括号内容
        kw = re.sub(r'\s+', '', kw)           # 移除空格
        return kw.strip()

class KeywordGenerator:
    def __init__(self, model_path):
        self.pipe = pipeline('text-generation', 
                           model=model_path,
                           tokenizer=model_path,
                           device=0 if torch.cuda.is_available() else -1)
    
    def generate(self, category, existing_kws):
        """智能生成关键词"""
        prompt = self._build_prompt(category, existing_kws)
        response = self.pipe(prompt, 
                           max_length=600,
                           num_return_sequences=1,
                           temperature=0.7,
                           top_p=0.9)
        return self._parse_response(response[0]['generated_text'])

    def _build_prompt(self, category, existing_kws):
        """构建动态提示模板"""
        example = {
            "卫星互联网": "星间激光通信, 卫星物联网, 高通量卫星...",
            "生物制造": "生物组装技术, 酶催化合成, 微生物燃料电池..."
        }.get(category, "示例关键词1, 示例关键词2, 示例关键词3...")

        return f"""请为【{category}】生成专业术语，要求：
1. 严格排除现有词汇：{', '.join(list(existing_kws)[:3])}...
2. 生成35个中文逗号分隔的新术语
3. 避免出现括号内容
4. 示例格式：{example}

请直接输出术语列表："""

    def _parse_response(self, text):
        """鲁棒性解析方法"""
        # 寻找最长的连续中文逗号序列
        matches = re.findall(r'([\u4e00-\u9fa5]+(?:[，,][\u4e00-\u9fa5]+){5,})', text)
        if matches:
            longest = max(matches, key=len)
            return [kw.strip() for kw in re.split(r'[，,、]', longest)]
        return []

def main(seed_file):
    # 初始化增强图谱
    graph = EnhancedKnowledgeGraph(seed_file)
    generator = KeywordGenerator("lora_rag_LLM/final_models")
    
    # 进度监控
    output_dir = "model/data"
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "generation_log.txt")

    while True:
        current_cat = graph.get_next_category()
        if not current_cat:
            break

        # 获取现有词汇
        existing = graph.nodes[current_cat]['keywords']
        if len(existing) >= 30:
            continue

        # 生成候选词
        candidates = generator.generate(current_cat, existing)
        valid_kws = graph.update_state(current_cat, candidates)

        # 保存结果
        with open(log_file, 'a', encoding='utf-8') as f:
            status = f"[{current_cat}] 新增{len(valid_kws)}词｜剩余需求：{graph.nodes[current_cat]['remaining']}"
            f.write(f"{status}\n新增词汇：{', '.join(valid_kws)}\n\n")

        print(status)

if __name__ == "__main__":
    seed_file = "model/data/balanced_keywords.txt"  # 替换为实际路径
    main(seed_file)