import os
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from rag import KnowMoveRAG  # 导入RAG模块
rag = KnowMoveRAG()
# 加载模型和分词器
final_save_path = "lora_rag_LLM/final_models"
model = AutoModelForCausalLM.from_pretrained(final_save_path)
tokenizer = AutoTokenizer.from_pretrained(final_save_path)

# 构建推理管道
pipe = pipeline('text-generation', model=model, tokenizer=tokenizer)

# 定义生成提示
original_prompt = """请按以下要求生成科技领域的专业关键词：
1. 按指定格式输出，类别前严格使用一个#，使用中文逗号分隔关键词
2. 每个类别生成90个专业关键词
3. 避免重复，使用中文逗号分隔

示例格式：
#卫星互联网
低轨卫星, 星间链路, 相控阵天线...

#生物制造
合成生物学, 细胞工厂...

请生成以下类别的关键词：
 智能制造与装备,新材料科技,生命健康技术,新能源与动力,传感器与仪器,数字智能技术

"""

enhanced_prompt = rag.augment_prompt(original_prompt)
# 定义必要的类别列表
required_categories = [
        "智能制造与装备", "新材料科技", "生命健康技术","新能源与动力", "传感器与仪器","数字智能技术"
]

def check_generated_text(text, required_categories):
    # 使用正则表达式匹配所有类别块
    pattern = re.compile(r'#([^\n]+)\n([^#]*)', re.DOTALL)
    matches = pattern.findall(text)
    
    categories = {}
    for category, keywords_part in matches:
        category = category.strip()
        # 清理关键词部分并分割
        keywords_part = re.sub(r'…+|\.\.\.+|\s+', '', keywords_part.strip())
        keywords = [k.strip() for k in keywords_part.split('，') if k.strip()]
        categories[category] = keywords
    
    # 检查是否包含所有必须的类别
    missing = set(required_categories) - set(categories.keys())
    if missing:
        print(f"错误：缺少以下类别：{missing}")
        return False
    
    # 检查每个类别的关键词数量及重复项
    for cat in required_categories:
        keywords = categories.get(cat, [])
        if not (1 <= len(keywords) <= 45):
            print(f"错误：类别 '{cat}' 的关键词数量为 {len(keywords)}，不符合要求（10-45个）")
            return False
        if len(keywords) != len(set(keywords)):
            print(f"错误：类别 '{cat}' 中存在重复关键词")
            return False
    return True

# 循环请求模型生成文本，直到格式检查通过
while True:
    # 生成文本
    generated_text = pipe(enhanced_prompt, max_length=5000, num_return_sequences=1)
    text = generated_text[0]["generated_text"]
    print("开始回答-------", text)
    
    # 检查格式
    if check_generated_text(text, required_categories):
        # 保存文件
        output_dir = os.path.join("model", "data")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "keywords.txt")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"文件已成功保存至：{output_path}")
        break
    else:
        print("格式检查未通过，重新生成中...")


