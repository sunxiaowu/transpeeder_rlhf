import os
import torch



if __name__ == "__main__":
    a1 = torch.load("/platform_tech/sunshuanglong/outputs/20240417_qinshu_step1_qwen1_5_14b_13000_v3_add_special_token_sp_20000/global_step336/layer_00-model_00-sequence_00-model_states.pt")
    a2 = torch.load("/platform_tech/sunshuanglong/outputs/20240417_qinshu_step1_qwen1_5_14b_13000_v3_add_special_token_sp_20000/global_step336/layer_00-model_00-sequence_01-model_states.pt")
    # 验证的layer_00-model_00-sequence_00-model_states.pt 和 layer_00-model_00-sequence_01-model_states.pt 不相等，也就是说，他们的loss计算没有同步更新