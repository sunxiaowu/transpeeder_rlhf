import json
from pprint import pprint
from pathlib import Path
from typing import Optional, Literal
from dataclasses import dataclass, field

import transformers
import pandas as pd
from tqdm import tqdm

def _chunk(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i: i + n]


@dataclass
class Arguments:
    # tokenizer_name_or_path: str = field(default="/platform_tech/models/deepseek-llm-7b-base-ckpt")
    tokenizer_name_or_path: str = field(default="/platform_tech/share_data/sunshuanglong/models/Qwen1.5-0.5B-Chat-ckpt")
    # data_path: str = field(default="/home/sunshuanglong/transpeeder/workspace/mulu_extract/data/train_type1_v3_step二阶段_sun_20240508二次优化.jsonl")
    # out_path: str = field(default="/home/sunshuanglong/transpeeder/workspace/mulu_extract/data/train_type1_v3_step二阶段_sun_20240508二次优化_check.jsonl")
    data_path: str = field(default="/home/sunshuanglong/transpeeder/workspace/qinshu-step1-qwen1_5/data/data_sftqingshu_v2_0407.jsonl")
    out_path: str = field(default="/home/sunshuanglong/transpeeder/workspace/qinshu-step1-qwen1_5/data/data_sftqingshu_v2_0407_1024_check.jsonl")
    max_seq_len: int = field(default=1024)

def main():
    parser = transformers.HfArgumentParser((Arguments,))
    args, = parser.parse_args_into_dataclasses()

    # tokenizer = transformers.AutoTokenizer.from_pretrained(args.tokenizer_name_or_path, use_fast=False)  
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.tokenizer_name_or_path)  # deepseek use_fast=False会报错

    def _count_tokens(texts):
        inputs = tokenizer.batch_encode_plus(texts)
        input_ids = inputs["input_ids"]
        return [len(x) for x in input_ids]

    token_nums = []
    # batch_size = 256 # 1 # 256
    cnt = 0
    writer = open(args.out_path, "w")

    with open(args.data_path, encoding="utf-8") as f:
        buff = []
        for ln in tqdm(f):
            try:
                j = json.loads(ln)
            except Exception as e:
                print(ln)
                continue
            text = j["prompt"] + j["output"]
            if len(j["prompt"].strip()) == 0 or len(j["output"].strip()) == 0:
                # print(j)
                cnt += 1
                continue
            if "openai" in j["output"].lower() or "chatgpt" in j["output"].lower() or "meta" in j["output"].lower() or "gpt3.5" in j["output"].lower():
                cnt += 1
                continue

            tmp = _count_tokens([text])
            if tmp[0] < args.max_seq_len:
                writer.write(json.dumps(j, ensure_ascii=False)+"\n")
                token_nums.extend(tmp)
            else:
                cnt += 1

    writer.close()
    print("drops {} samples".format(cnt))

    df = pd.DataFrame({'token_num': token_nums})
    desc = df.describe(
        percentiles=[.5, .75, .85, .90, .95],
    )
    pprint(desc)


if __name__ == "__main__":
    main()
