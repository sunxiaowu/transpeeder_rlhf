import os
import json
import random
from tqdm import tqdm

def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        content = json.loads(f.read().strip())
    return content

if __name__ == "__main__":
    res_list = list()
    res_list_short = list()
    json_lines = load_json("/home/sunshuanglong/transpeeder/data/alpaca_gpt4_data_zh.json")
    num = len(json_lines)
    # random.shuffle(json_lines)
    # b = 3
    for json_item in tqdm(json_lines[:num]):
        tmp = len(json_item["instruction"])
        if len(json_item["instruction"]) < 50:
            # continue
            res_list_short.append(json.dumps({"prompt":json_item["instruction"] + json_item["input"], "output":json_item["output"]}, ensure_ascii=False))
        else:
            res_list.append(json.dumps({"prompt":json_item["instruction"] + json_item["input"], "output":json_item["output"]}, ensure_ascii=False))
    alpaca_all = res_list + res_list_short[:5000]
    random.shuffle(alpaca_all)
    with open(f"/home/sunshuanglong/transpeeder/data/alpaca_gpt_data_zh_choose_{len(alpaca_all)}.jsonl", "w", encoding="utf-8") as f:
        f.write("\n".join(alpaca_all))
    print(f"len:{len(alpaca_all)}")