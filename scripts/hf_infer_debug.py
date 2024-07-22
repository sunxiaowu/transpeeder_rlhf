import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import transformers

model_name = "/platform_tech/sunshuanglong/saved_models/workspace/security-20240206-7b/output/security-20240206-deepseek-llm-7b-add-token3/global_191_merge"
# tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="right")

tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name,
        model_max_length=1024,
        padding_side="right",
        # use_fast=False,
    )

model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
model.generation_config = GenerationConfig.from_pretrained(model_name)
model.generation_config.pad_token_id = model.generation_config.eos_token_id

text = "An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. The output is"
text = """你是一个文本分类器。已知分类列表:['伦理道德', '写审稿意见', '不满意-不明原因', '隐私和财产',  \
   '询问机器人身份', '研究方法', '深入讨论-进一步启示', '论文综述', '客户投诉', '用户建议', '论文推荐', \
   '研究结论', '研究贡献', '其他', '侮辱', '研究问题', '目标劫持', '深入讨论-讨论优缺点', '分段概述', \
   '信息提取-实验部分', '信息提取-', '表示不清楚或不知道', '公式解析', '身体伤害', '局部总结', '图表解析', \
   '概念解释', '无意义', '研究背景', '深入讨论', '文章创新与不足', 'Prompt泄漏', '不合理/不安全的指令', '分段总结', \
   '全文总结', '反向曝光', '表示不理解', '犯罪和非法活动', '心理健康', '于角色扮演攻击的指令', '隐藏不安全观点的询问', '偏见与歧视']。 \
请你根据文本内容做选择题，从分类列表选择一个最合适的分类选项，并输出。输出结果只需要输出分类选项,无需解释以及多余的回答。如果分类列表均不合适，输出文本‘其他’。\n\n文本内容：这篇论文讲了啥\n分类标签："""

inputs = tokenizer(text, return_tensors="pt")
outputs = model.generate(**inputs.to(model.device), max_new_tokens=10)

result = tokenizer.decode(outputs[0], skip_special_tokens=False)
print(result)