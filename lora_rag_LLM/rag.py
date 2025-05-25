import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

class KnowMoveRAG:
    def __init__(self, data_folder="knowmove"):
        self.prompts, self.completions = self._load_category_knowledge(data_folder)
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 3),
            stop_words=["请生成", "专业关键词", "领域"]  # 适配新prompt格式的停用词
        )
        self._precompute_tfidf()

    def _load_category_knowledge(self, folder):
        """加载按类别组织的知识"""
        prompts, completions = [], []
        
        # 匹配新文件名格式：keyword_类别.txt
        for fname in os.listdir(folder):
            if fname.startswith("keyword_") and fname.endswith(".txt"):
                category = fname[8:-4]  # 提取"高端装备制造"等类别名
                file_path = os.path.join(folder, fname)
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = [line.strip() for line in f.readlines()]
                    
                    # 解析新格式
                    prompt = next(line[len("prompt: "):] for line in lines 
                                if line.startswith("prompt: "))
                    completion = next(line[len("completion："):] for line in lines 
                                    if line.startswith("completion："))
                    
                    # 标准化prompt格式
                    std_prompt = f"生成{category}领域的关键词"
                    prompts.append(std_prompt)
                    completions.append(completion)
        
        return prompts, completions

    def _precompute_tfidf(self):
        if len(self.prompts) == 0:
            raise ValueError("知识库为空，请检查knowmove文件夹")
        self.tfidf_matrix = self.vectorizer.fit_transform(self.prompts)

    def retrieve(self, query, topk=3):
        """支持多类别查询的检索"""
        if self.tfidf_matrix is None:
            return []
            
        # 提取查询中的领域关键词
        detected_categories = [
            cat for cat in self.get_all_categories() 
            if cat in query
        ]
        
        # 如果没有检测到领域，则使用TF-IDF检索
        if not detected_categories:
            return self._tfidf_retrieve(query, topk)
            
        # 优先返回匹配领域的知识
        return self._category_based_retrieve(detected_categories, topk)

    def _tfidf_retrieve(self, query, topk):
        """原始TF-IDF检索"""
        query_vec = self.vectorizer.transform([query])
        sim_scores = linear_kernel(query_vec, self.tfidf_matrix).flatten()
        top_indices = np.argsort(-sim_scores)[:topk]
        return [(self.prompts[i], self.completions[i]) for i in top_indices]

    def _category_based_retrieve(self, categories, topk):
        """基于领域名称的检索"""
        category_prompts = {
            p: c for p, c in zip(self.prompts, self.completions)
            if any(cat in p for cat in categories)
        }
        return list(category_prompts.items())[:topk]

    def get_all_categories(self):
        """从prompt中提取所有领域名称"""
        return list(set(
            p.split("生成")[1].split("领域")[0] 
            for p in self.prompts
        ))

    def augment_prompt(self, original_prompt):
        retrieved = self.retrieve(original_prompt)
        examples = []
        for p, c in retrieved:
            example = f"【{p}】\n{c.replace(', ', '、')}"  # 统一使用中文顿号
            examples.append(example)
        nl = '\n'  # 定义换行符变量
        return f'''请严格遵循以下示例格式（保留井号分类和换行）：
{nl.join(examples)}

当前任务：
{original_prompt}'''