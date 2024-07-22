import os
import json
import argparse
from pathlib import Path

import torch


PARAM_MAP = {
    "6B": {
        "n_layers": 32,
    },
    "7B": {
        "n_layers": 32,
    },
    "7B-deepseek": {
        "n_layers": 30,
    },
    "13B": {
        "n_layers": 40,
    },
    "30B": {
        "n_layers": 60,
    },
    "34B": {
        "n_layers": 60,
    },
    "65B": {
        "n_layers": 80,
    },
}


def read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def write_json(text, path):
    with open(path, "w") as f:
        json.dump(text, f)


def write_model(model_path, input_base_path, model_size, tokenizer_size):
    assert model_size in PARAM_MAP
    os.makedirs(model_path, exist_ok=True)

    params = PARAM_MAP[model_size]
    n_layers = params["n_layers"]

    loaded = {}
    ORIGINAL_TOKENIZER_SIZE = tokenizer_size
    for pt in Path(input_base_path).iterdir():
        # assert tp/mp == 1
        if not pt.name.startswith('layer_'):
            continue
        sd = torch.load(pt, map_location="cpu")
        
        if pt.name == 'layer_00-model_00-model_states.pt':
            loaded['model.embed_tokens.weight'] = sd['weight'][: ORIGINAL_TOKENIZER_SIZE, :]
            continue
        if pt.name == f'layer_{n_layers + 1}-model_00-model_states.pt':
            loaded['model.norm.weight'] = sd['weight']
            continue
        if pt.name == f'layer_{n_layers + 2}-model_00-model_states.pt':
            loaded['lm_head.weight'] = sd['weight'][: ORIGINAL_TOKENIZER_SIZE, :]
            continue

        layer_i = int(pt.name.split('-')[0].replace('layer_', '')) - 1
        layer_sd = { f"model.layers.{layer_i}.{nm}": weight for nm, weight in sd.items() }
        loaded.update(layer_sd)


    torch.save(loaded, os.path.join(model_path, "pytorch_model.bin"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        # default="/platform_tech/sunshuanglong/saved_models/workspace/scim-1207/output/1207-scim-200/global_step150",
        default="/platform_tech/models/sft_model/workspace/mulu/global_step100",
        help="Location of LLaMA weights, which contains tokenizer.model and model folders",
    )
    parser.add_argument(
        "--model_size",
        default="6B",
        choices=["7B-deepseek", "7B", "13B", "30B", "34B", "65B"],
    )
    parser.add_argument(
        "--output_dir",
        default="/platform_tech/models/sft_model/workspace/mulu/global_step100_merge",
        help="Location to write HF model and tokenizer",
    )
    parser.add_argument(
    "--tokenizer_size",
    default="64000",
    help="Size of tokenizer",
    type=int,
    )
    args = parser.parse_args()
    write_model(
        model_path=args.output_dir,
        input_base_path=args.input_dir,
        model_size=args.model_size,
        tokenizer_size=args.tokenizer_size,
    )


if __name__ == "__main__":
    main()
