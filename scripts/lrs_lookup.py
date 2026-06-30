import json 
import os 
import hashlib
current_dir = os.path.dirname(os.path.abspath(__file__))


with open(os.path.join(current_dir, "lrs/dpo.json"), "r") as f:
    dpo_lrs = json.load(f)

with open(os.path.join(current_dir, "lrs/grpo.json"), "r") as f:
    grpo_lrs = json.load(f)

with open(os.path.join(current_dir, "lrs/instruct.json"), "r") as f:
    instruct_lrs = json.load(f)

with open(os.path.join(current_dir, "lrs/grpo_python.json"), "r") as f:
    grpo_python_lrs = json.load(f)


def hash_model(model: str) -> str:
    model_bytes = model.encode('utf-8')
    hashed = hashlib.sha256(model_bytes).hexdigest()
    return hashed 


def get_dpo_lr(model: str):
    hashed_model = hash_model(model)
    for lr in dpo_lrs:
        if lr["h"] == hashed_model:
            return lr["lr"]
    return None


def get_grpo_lr(model: str):
    hashed_model = hash_model(model)
    for lr in grpo_lrs:
        if lr["h"] == hashed_model:
            return lr["lr"]
    return None

def get_instruct_lr(model: str):
    hashed_model = hash_model(model)
    for lr in instruct_lrs:
        if lr["h"] == hashed_model:
            return lr["lr"]
    return None


def get_grpo_python_lr(model: str):
    hashed_model = hash_model(model)
    for lr in grpo_python_lrs:
        if lr["h"] == hashed_model:
            return lr["lr"]
    return None


def _read_csv_ar(path: str):
    with open(path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]
    result = []
    for line in lines[1:]:  # skip header
        parts = line.split(",")
        result.append({
            "size": int(parts[0]),
            "ar": parts[1].strip().lower(),
            "lr": float(parts[2]),
        })
    return result


INSTRUCT_AR_CONFIG = _read_csv_ar(os.path.join(current_dir, "lrs/instruct_ar.csv"))
DPO_AR_CONFIG = _read_csv_ar(os.path.join(current_dir, "lrs/dpo_ar.csv"))


def _get_lr_from_ar(architecture: str, param_nums: int, configs: list):
    filtered = [c for c in configs if c["ar"] == architecture.strip().lower()]
    if not filtered:
        return None
    closest = min(filtered, key=lambda c: abs(c["size"] - param_nums))
    print(f"Using lr from architecture config: {closest['lr']} (arch={architecture}, size={param_nums})", flush=True)
    return closest["lr"]


def get_lr_from_ar_instruct(architecture: str, param_nums: int):
    return _get_lr_from_ar(architecture, param_nums, INSTRUCT_AR_CONFIG)


def get_lr_from_ar_dpo(architecture: str, param_nums: int):
    return _get_lr_from_ar(architecture, param_nums, DPO_AR_CONFIG)
