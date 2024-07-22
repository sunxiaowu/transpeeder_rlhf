from pathlib import Path
from typing import Optional, Literal
from dataclasses import dataclass, field

import torch
import transformers

from transpeeder.models.patching import (
    smart_tokenizer_and_embedding_resize,
)
from transpeeder.feeder import (
    DEFAULT_BOS_TOKEN,
    DEFAULT_PAD_TOKEN,
    DEFAULT_EOS_TOKEN,
    DEFAULT_UNK_TOKEN,
)


@dataclass
class Arguments:
    # model_name_or_path: Optional[str] = field(default="/platform_tech/models/deepseek-llm-7b-base")
    # tokenizer_path: Optional[str] = field(default="/platform_tech/models/deepseek-llm-7b-base")
    # output_dir: str = field(default="/platform_tech/models/deepseek-llm-7b-base-ckpt2")
    model_name_or_path: Optional[str] = field(default="/platform_tech/models/deepseek-coder-33b-base")
    tokenizer_path: Optional[str] = field(default="/platform_tech/models/deepseek-coder-33b-base")
    output_dir: str = field(default="/platform_tech/models/deepseek-coder-33b-base-ckpt")
    # mp_world_size: int = field(default=8)
    mp_world_size: int = field(default=4)
    add_special_tokens: Optional[bool] = field(default=False)

# 仅对llama结构
def write_ckpt(outpath: Path, model: torch.nn.Module, model_config: transformers.AutoConfig, mp: int):
    loaded = model.state_dict()
    n_layers = model_config.num_hidden_layers

    # embedding
    sd = {"weight": loaded['model.embed_tokens.weight']}
    torch.save(sd, outpath / "layer_00-model_00-model_states.pt")
    # norm
    sd = {f"weight": loaded['model.norm.weight']}
    torch.save(sd, outpath / f"layer_{n_layers + 1}-model_00-model_states.pt")
    # lm head
    sd = {f"weight": loaded['lm_head.weight']}
    torch.save(sd, outpath / f"layer_{n_layers + 2}-model_00-model_states.pt")
    # decoder layers
    for layer_i in range(n_layers):
        sd = {nm.replace(f"model.layers.{layer_i}.", f""): weight for nm, weight in loaded.items() if nm.startswith(f"model.layers.{layer_i}.") and not nm.endswith("inv_freq")}
        torch.save(sd, outpath / f"layer_{layer_i + 1:02d}-model_00-model_states.pt")

    model_state = {
        "dp_world_size": 1,
        "mp_world_size": mp,
        "module": None,
        "optimizer": None,
        "global_steps": 1,
        "skipped_steps": 1,
        "iteration": 1,
        
    }
    for rank in range(mp):
        torch.save(model_state, outpath / f"mp_rank_{rank:02d}_model_states.pt")


def main():
    print("convert model to ckpt...")
    parser = transformers.HfArgumentParser((Arguments,))
    args, = parser.parse_args_into_dataclasses()

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.tokenizer_path)
    model_config = transformers.AutoConfig.from_pretrained(args.model_name_or_path)
    model = transformers.AutoModelForCausalLM.from_pretrained(args.model_name_or_path)

    print("finish loading...")

    if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
        )

    if "llama" in args.model_name_or_path: 
        tokenizer.add_special_tokens(
            {
                "eos_token": DEFAULT_EOS_TOKEN,
                "bos_token": DEFAULT_BOS_TOKEN,
                "unk_token": DEFAULT_UNK_TOKEN,
            }
        )

    if args.add_special_tokens:
        # "[START_REF]","[END_REF]",
        special_tokens_map = {
            "additional_special_tokens": ["<|prefix_begin|>", "<|prefix_end|>", "<|prompter|>", "<|endoftext|>", "<|assistant|>", "[START_REF]", "[END_REF]"]
        }

        tokenizer.add_special_tokens(special_tokens_map)

        smart_tokenizer_and_embedding_resize(
                    special_tokens_dict=special_tokens_map,
                    tokenizer=tokenizer,
                    model=model,
                )
    
    print(len(tokenizer))
    print(model.get_input_embeddings().weight.shape[0])

    outpath = Path(args.output_dir)
    if outpath.exists():
        print(f"{outpath} exists. Do nothing.")
        exit(0)

    print(f"create {outpath}")
    outpath.mkdir()
    steppath = outpath / "global_step001"
    steppath.mkdir()

    write_ckpt(steppath, model, model_config, args.mp_world_size)
    with open(outpath / "latest", "w") as fout:
        fout.write("global_step001")

    tokenizer.save_pretrained(outpath)
    model_config.save_pretrained(outpath)


if __name__ == "__main__":
    main()
