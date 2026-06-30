from transformers import AutoConfig, AutoTokenizer
from safetensors.torch import load_file
import glob
import torch
import os
import json


def is_reasoning_tokenizer(tokenizer: AutoTokenizer) -> bool:
    try:
        vocab = tokenizer.get_vocab()
        
        pairs = [
            ('<think>', '</think>'),
            ('<thinking>', '</thinking>'),
            ('<reasoning>', '</reasoning>'),
            ('<thought>', '</thought>'),
            ('<reflection>', '</reflection>'),
        ]
        
        return any(open_tag in vocab and close_tag in vocab 
                   for open_tag, close_tag in pairs)
    except:
        return False


def get_model_architecture(model_path: str) -> str:
    try:
        config = AutoConfig.from_pretrained(model_path)
        architectures = config.architectures
        if len(architectures) > 1:
            return "Multiple architectures"
        return architectures[0].strip().lower()
    except:
        return "Unknown"


def get_use_liger(architecture: str) -> str:
    if architecture.lower() in [
        "qwen2forcausallm",
        "llamaforcausallm",
        "gemma2forcausallm",
        "mixtralforcausallm",
        "mistralforcausallm",
        "qwen3forcausallm",
        "phi3forcausallm",
        "gemmaforcausallm",
    ]:
        return "True"
    else:
        return "False"


def count_params_from_safetensors(model_dir):
    total_params = 0
    shards = glob.glob(os.path.join(model_dir, "*.safetensors"))
    if not shards:
        return None

    for shard_path in shards:
        print(f"Loading shard: {shard_path}")
        tensors = load_file(shard_path)
        total_params += sum(v.numel() for v in tensors.values())

    return total_params


def count_params_from_bin(model_dir):
    total_params = 0
    shards = glob.glob(os.path.join(model_dir, "*.bin"))
    if not shards:
        return None

    for shard_path in shards:
        print(f"Loading shard: {shard_path}")
        try:
            state_dict = torch.load(shard_path, map_location="cpu")
            total_params += sum(v.numel() for v in state_dict.values())
        except Exception as e:
            print(f"cannot load {shard_path}: {e}")
            continue

    return total_params


def get_model_size_from_local_path(model_path: str) -> int:
    size = count_params_from_safetensors(model_path)
    if size is not None and size > 1000:
        print(f"Model size from safetensors: {size}")
        return size
    size = count_params_from_bin(model_path)
    if size is not None and size > 1000:
        print(f"Model size from bin: {size}")
        return size
    return None


def get_gpu_count():
    return torch.cuda.device_count()


def get_model_num_params(model_path: str) -> int:
    size = get_model_size_from_local_path(model_path)
    if size is not None:
        return size
    print(f"Cannot determine model size for {model_path}, returning None")
    return None


def disable_flash_attention(architecture: str) -> str:
    if architecture.strip().lower() in [
        "gptneoforcausallm",
        "bloomforcausallm",
        "gptossforcausallm",
        "phiforcausallm",     # phi-2: PhiForCausalLM
        "falconforcausallm",  # falcon-rw variants: FalconForCausalLM
    ]:
        return "True"
    return "False"


def disable_action_mask(model_path: str) -> str:
    # LlamaTokenizer (non-Fast) is used by Mistral v0.2/v0.3 and CodeLlama;
    # LlamaTokenizerFast with legacy=True is used by deepseek-coder.
    # All cause action mask misalignment due to BPE byte-level tokenization quirks.
    try:
        tokenizer_config_path = os.path.join(model_path, "tokenizer_config.json")
        with open(tokenizer_config_path, "r") as f:
            tokenizer_config = json.load(f)
        tokenizer_class = tokenizer_config.get("tokenizer_class", "")
        if tokenizer_class in ("LlamaTokenizer", "CodeLlamaTokenizer"):
            return "True"
        if tokenizer_class == "LlamaTokenizerFast" and tokenizer_config.get("legacy") is True:
            return "True"
    except Exception as e:
        print(f"Could not read tokenizer_config.json from {model_path}: {e}")
    return "False"


def get_use_vllm(architecture: str) -> str:
    if architecture.strip().lower() in [
        "gptneoforcausallm",
        "bloomforcausallm",
        "falconforcausallm",  # falcon-rw variants: FalconForCausalLM
        "phiforcausallm",     # phi-2: PhiForCausalLM
        "gptjforcausallm",
    ]:
        return False
    return True


def get_gradient_checkpointing(architecture: str) -> str:
    # FalconForCausalLM does not support gradient checkpointing
    if architecture.strip().lower() == "falconforcausallm":
        return "False"
    return "True"


def get_data_size(data_path: str) -> int:
    with open(data_path, "r") as f:
        data = json.load(f)
    return len(data)
