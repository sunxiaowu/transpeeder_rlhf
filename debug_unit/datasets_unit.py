from datasets import load_dataset
from tqdm import tqdm
import json

def read_dataset(data_path):
    return load_dataset(data_path)

if __name__ == "__main__":
    res_dict_train = list()
    res_dict_dev = list()
    data_path = "/platform_tech/sunshuanglong/models/rm-static/data"
    rm_output_path = "/home/sunshuanglong/transpeeder/data"

    data_info = read_dataset(data_path)
    a = 3

    train_len = data_info["train"].num_rows
    dev_len = data_info["test"].num_rows

    # debug
    train_len = 300
    dev_len = 100

    for _ in tqdm(range(train_len), desc="train data prepare..."):
        item_dict = {"prompt":data_info["train"]["prompt"][_], "output":"", "response":data_info["train"]["response"][_], "chosen":data_info["train"]["chosen"][_], "rejected":data_info["train"]["rejected"][_]}
        res_dict_train.append(json.dumps(item_dict, ensure_ascii=False))
    print("data train trans done~~")

    for _ in tqdm(range(dev_len), desc="test data prepare..."):
        item_dict = {"prompt":data_info["test"]["prompt"][_], "output":"", "response":data_info["test"]["response"][_], "chosen":data_info["train"]["chosen"][_], "rejected":data_info["test"]["rejected"][_]}
        res_dict_dev.append(json.dumps(item_dict, ensure_ascii=False))
    print("data dev trans done~~")

    with open(rm_output_path + f"/rm_static_train_{len(res_dict_train)}.jsonl", "w", encoding="utf-8") as f:
        f.write("\n".join(res_dict_train))
    print("data train write done~~")
    with open(rm_output_path + f"/rm_static_test_{len(res_dict_dev)}.jsonl", "w", encoding="utf-8") as f:
        f.write("\n".join(res_dict_dev))
    print("data dev write done~")
    
