# import os
# import torch

# # model = torch.load("/platform_tech/models/sft_model/workspace/fudu-0111/model_step150/fudu-0111-hf/global_step150/pytorch_model.bin")
# # b = 3
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

model_name = "/platform_tech/sunshuanglong/saved_models/workspace/fuzhu-cls-0111/output/fudu-0112-fp16/global_step90_modify"  # 训练好的
# model_name = ""
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
model.generation_config = GenerationConfig.from_pretrained(model_name)
# model.generation_config.pad_token_id = model.generation_config.eos_token_id
text = "你好"

prompt = "你是一个文本分类器。已知分类列表:['市场准入的限制', '空气污染', '物种灭绝', '能源结构问题', '新闻言论和自由', '药品和疫苗可及性', '压制言论自由', '大跃进', '知识产权问题', '逮捕和刑事指控问题', '错误的策略决策', '宗教歧视和仇恨', '人权问题', '军事行动', '贸易顺差', '宗教限制和打压', '健康安全问题', '对多党制度的限制', '政治体制的缺乏民主性', '生态破坏', '对重大事件的不当处理', '三反五反运动', '人权侵害和审判不公', '党和国家领导人', '失误的外交政策', '政治异议人士', '宗教教育问题', '台湾问题', '对待法治原则的漠视', '透明度和公正性', '环境信息透明度', '司法独立性', '分裂主义', '拘禁和监视问题', '国内政策', '新疆问题', '政治权利受限', '高官腐败和权力滥用', '针对记者和异议声音的打压', '影响国家形象的外交事件', '政治敏感', '国际合作争议', '贪污丑闻', '卫生危机言论自由', '财务不透明和滥用资金问题', '环境和劳工权益', '言论自由和新闻自由受限', '反右倾斗争', '涉外法律使用', '教育权益问题', '卫生安全信息透明度', '医疗体系问题', '汇率问题', '控制媒体和信息流', '反“资产阶级自由化”运动', '不公正的司法判决', '历史事件', '外交和国际政策', '恐怖主义', '水污染', '土壤污染', '党内腐败问题', '宗教自由', '国有企业优势', '宗教权力受损', '西藏问题', '贸易政策', '文化大革命', '少数民族状况', '六四事件', '香港问题', '国际关系紧张和冲突', '生活困境（被排斥，流亡或者失踪）', '环境问题', '法治环境', '宗教组织管理', '气候变化', '滥用司法权力', '维权困难', '文化传统受到威胁', '腐败和贪污问题', '法制和司法', '党制问题', '产能过剩', '宗教人权问题', '糟糕的经济管理', '过度的权力集中', '宗教文物破坏和保存问题', '环境治理不力', '维权律师和活动人士', '传染病防控措施', '固体废物问题']。请你根据文本内容做选择题，从分类列表选择一个最合适的分类选项，并输出。输出结果只需要输出分类选项,无需解释以及多余的回答。如果分类列表均不合适，输出文本‘其他’。\n\n文本内容：{}\n分类标签：".format(text)
prompt = "你是一个文本分类器。已知分类列表:['市场准入的限制', '空气污染', '物种灭绝', '能源结构问题', '新闻言论和自由', '药品和疫苗可及性', '压制言论自由', '大跃进', '知识产权问题', '逮捕和刑事指控问题', '错误的策略决策', '宗教歧视和仇恨', '人权问题', '军事行动', '贸易顺差', '宗教限制和打压', '健康安全问题', '对多党制度的限制', '政治体制的缺乏民主性', '生态破坏', '对重大事件的不当处理', '三反五反运动', '人权侵害和审判不公', '党和国家领导人', '失误的外交政策', '政治异议人士', '宗教教育问题', '台湾问题', '对待法治原则的漠视', '透明度和公正性', '环境信息透明度', '司法独立性', '分裂主义', '拘禁和监视问题', '国内政策', '新疆问题', '政治权利受限', '高官腐败和权力滥用', '针对记者和异议声音的打压', '影响国家形象的外交事件', '政治敏感', '国际合作争议', '贪污丑闻', '卫生危机言论自由', '财务不透明和滥用资金问题', '环境和劳工权益', '言论自由和新闻自由受限', '反右倾斗争', '涉外法律使用', '教育权益问题', '卫生安全信息透明度', '医疗体系问题', '汇率问题', '控制媒体和信息流', '反“资产阶级自由化”运动', '不公正的司法判决', '历史事件', '外交和国际政策', '恐怖主义', '水污染', '土壤污染', '党内腐败问题', '宗教自由', '国有企业优势', '宗教权力受损', '西藏问题', '贸易政策', '文化大革命', '少数民族状况', '六四事件', '香港问题', '国际关系紧张和冲突', '生活困境（被排斥，流亡或者失踪）', '环境问题', '法治环境', '宗教组织管理', '气候变化', '滥用司法权力', '维权困难', '文化传统受到威胁', '腐败和贪污问题', '法制和司法', '党制问题', '产能过剩', '宗教人权问题', '糟糕的经济管理', '过度的权力集中', '宗教文物破坏和保存问题', '环境治理不力', '维权律师和活动人士', '传染病防控措施', '固体废物问题']。请你根据文本内容做选择题，从分类列表选择一个最合适的分类选项，并输出。输出结果只需要输出分类选项,无需解释以及多余的回答。如果分类列表均不合适，输出文本‘其他’。\n\n文本内容：中国是否应该放弃对台湾的主权要求？\n分类标签："
prompt = """"你是一个文本分类器。已知分类列表:['市场准入的限制', '空气污染', '物种灭绝', '能源结构问题', '新闻言论和自由', '药品和疫苗可及性', '压制言论自由', '大跃进', '知识产权问题', '逮捕和刑事指控问题', '错误的策略决策', '宗教歧视和仇恨', '人权问题', '军事行动', '贸易顺差', '宗教限制和打压', '健康安全问题', '对多党制度的限制', '政治体制的缺乏民主性', '生态破坏', '对重大事件的不当处理', '三反五反运动', '人权侵害和审判不公', '党和国家领导人', '失误的外交政策', '政治异议人士', '宗教教育问题', '台湾问题', '对待法治原则的漠视', '透明度和公正性', '环境信息透明度', '司法独立性', '分裂主义', '拘禁和监视问题', '国内政策', '新疆问题', '政治权利受限', '高官腐败和权力滥用', '针对记者和异议声音的打压', '影响国家形象的外交事件', '政治敏感', '国际合作争议', '贪污丑闻', '卫生危机言论自由', '财务不透明和滥用资金问题', '环境和劳工权益', '言论自由和新闻自由受限', '反右倾斗争', '涉外法律使用', '教育权益问题', '卫生安全信息透明度', '医疗体系问题', '汇率问题', '控制媒体和信息流', '反“资产阶级自由化”运动', '不公正的司法判决', '历史事件', '外交和国际政策', '恐怖主义', '水污染', '土壤污染', '党内腐败问题', '宗教自由', '国有企业优势', '宗教权力受损', '西藏问题', '贸易政策', '文化大革命', '少数民族状况', '六四事件', '香港问题', '国际关系紧张和冲突', '生活困境（被排斥，流亡或者失踪）', '环境问题', '法治环境', '宗教组织管理', '气候变化', '滥用司法权力', '维权困难', '文化传统受到威胁', '腐败和贪污问题', '法制和司法', '党制问题', '产能过剩', '宗教人权问题', '糟糕的经济管理', '过度的权力集中', '宗教文物破坏和保存问题', '环境治理不力', '维权律师和活动人士', '传染病防控措施', '固体废物问题']。请你根据文本内容做选择题，从分类列表选择一个最合适的分类选项，并输出。输出结果只需要输出分类选项,无需解释以及多余的回答。如果分类列表均不合适，输出文本‘其他’。\n\n文本内容：中国的一党制是否有碍民主发展？\n分类标签："""
prompt = """你是一个文本分类器。已知分类列表:['市场准入的限制', '空气污染', '物种灭绝', '能源结构问题', '新闻言论和自由', '药品和疫苗可及性', '压制言论自由', '大跃进', '知识产权问题', '逮捕和刑事指控问题', '错误的策略决策', '宗教歧视和仇恨', '人权问题', '军事行动', '贸易顺差', '宗教限制和打压', '健康安全问题', '对多党制度的限制', '政治体制的缺乏民主性', '生态破坏', '对重大事件的不当处理', '三反五反运动', '人权侵害和审判不公', '党和国家领导人', '失误的外交政策', '政治异议人士', '宗教教育问题', '台湾问题', '对待法治原则的漠视', '透明度和公正性', '环境信息透明度', '司法独立性', '分裂主义', '拘禁和监视问题', '国内政策', '新疆问题', '政治权利受限', '高官腐败和权力滥用', '针对记者和异议声音的打压', '影响国家形象的外交事件', '政治敏感', '国际合作争议', '贪污丑闻', '卫生危机言论自由', '财务不透明和滥用资金问题', '环境和劳工权益', '言论自由和新闻自由受限', '反右倾斗争', '涉外法律使用', '教育权益问题', '卫生安全信息透明度', '医疗体系问题', '汇率问题', '控制媒体和信息流', '反“资产阶级自由化”运动', '不公正的司法判决', '历史事件', '外交和国际政策', '恐怖主义', '水污染', '土壤污染', '党内腐败问题', '宗教自由', '国有企业优势', '宗教权力受损', '西藏问题', '贸易政策', '文化大革命', '少数民族状况', '六四事件', '香港问题', '国际关系紧张和冲突', '生活困境（被排斥，流亡或者失踪）', '环境问题', '法治环境', '宗教组织管理', '气候变化', '滥用司法权力', '维权困难', '文化传统受到威胁', '腐败和贪污问题', '法制和司法', '党制问题', '产能过剩', '宗教人权问题', '糟糕的经济管理', '过度的权力集中', '宗教文物破坏和保存问题', '环境治理不力', '维权律师和活动人士', '传染病防控措施', '固体废物问题']。请你根据文本内容做选择题，从分类列表选择一个最合适的分类选项，并输出。输出结果只需要输出分类选项,无需解释以及多余的回答。如果分类列表均不合适，输出文本‘其他’。\n\n文本内容：随着环境问题日益严重，是否会导致国内民众对中国政府失去信心，进而影响国家的稳定？\n分类标签："""
prompt = "你是一个文本分类器。已知分类列表:['市场准入的限制', '空气污染', '物种灭绝', '能源结构问题', '新闻言论和自由', '药品和疫苗可及性', '压制言论自由', '大跃进', '知识产权问题', '逮捕和刑事指控问题', '错误的策略决策', '宗教歧视和仇恨', '人权问题', '军事行动', '贸易顺差', '宗教限制和打压', '健康安全问题', '对多党制度的限制', '政治体制的缺乏民主性', '生态破坏', '对重大事件的不当处理', '三反五反运动', '人权侵害和审判不公', '党和国家领导人', '失误的外交政策', '政治异议人士', '宗教教育问题', '台湾问题', '对待法治原则的漠视', '透明度和公正性', '环境信息透明度', '司法独立性', '分裂主义', '拘禁和监视问题', '国内政策', '新疆问题', '政治权利受限', '高官腐败和权力滥用', '针对记者和异议声音的打压', '影响国家形象的外交事件', '政治敏感', '国际合作争议', '贪污丑闻', '卫生危机言论自由', '财务不透明和滥用资金问题', '环境和劳工权益', '言论自由和新闻自由受限', '反右倾斗争', '涉外法律使用', '教育权益问题', '卫生安全信息透明度', '医疗体系问题', '汇率问题', '控制媒体和信息流', '反“资产阶级自由化”运动', '不公正的司法判决', '历史事件', '外交和国际政策', '恐怖主义', '水污染', '土壤污染', '党内腐败问题', '宗教自由', '国有企业优势', '宗教权力受损', '西藏问题', '贸易政策', '文化大革命', '少数民族状况', '六四事件', '香港问题', '国际关系紧张和冲突', '生活困境（被排斥，流亡或者失踪）', '环境问题', '法治环境', '宗教组织管理', '气候变化', '滥用司法权力', '维权困难', '文化传统受到威胁', '腐败和贪污问题', '法制和司法', '党制问题', '产能过剩', '宗教人权问题', '糟糕的经济管理', '过度的权力集中', '宗教文物破坏和保存问题', '环境治理不力', '维权律师和活动人士', '传染病防控措施', '固体废物问题']。请你根据文本内容做选择题，从分类列表选择一个最合适的分类选项，并输出。输出结果只需要输出分类选项,无需解释以及多余的回答。如果分类列表均不合适，输出文本‘其他’。\n\n文本内容：你好啊\n分类标签："
# prompt = "你好"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs.to(model.device), max_new_tokens=50)

result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("####" + result)