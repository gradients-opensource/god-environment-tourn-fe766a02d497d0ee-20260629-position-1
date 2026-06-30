"""
Orchestrator config for EnvTask SFT training.
Counterpart of instruct_config.py for the SFT environment mode.
No GRPO content; no vLLM/num_generations/beta.
"""

from copy import deepcopy

from envs.shared_env import _log
from lrs_lookup import get_lr_from_ar_instruct
from model_utility import (
    disable_flash_attention,
    get_gpu_count,
    get_model_architecture,
    get_model_num_params,
    get_use_liger,
)

SFT_ENV_SIZE_CONFIG: dict[str, dict] = {
    "0_1_b":  {"lr": 5e-5,   "distributed": "ddp", "gpu_count": 1, "batch_size": 32,  "gradient_accumulation_steps": 1, "use_lora": False},
    "1_2_b":  {"lr": 5e-5,   "distributed": "ddp", "gpu_count": 1, "batch_size": 32,  "gradient_accumulation_steps": 1, "use_lora": False},
    "2_4_b":  {"lr": 5e-5,   "distributed": "ddp", "gpu_count": 1, "batch_size": 24,  "gradient_accumulation_steps": 1, "use_lora": False},
    "4_5_b":  {"lr": 7e-5,   "distributed": "ddp", "gpu_count": 2, "batch_size": 32,  "gradient_accumulation_steps": 1, "use_lora": True},
    "5_9_b":  {"lr": 3.5e-5, "distributed": "ddp", "gpu_count": 2, "batch_size": 32,  "gradient_accumulation_steps": 1, "use_lora": True},
    "9_12_b": {"lr": 1e-4,   "distributed": "ddp", "gpu_count": 2, "batch_size": 32,  "gradient_accumulation_steps": 1, "use_lora": True},
    "12_15_b":{"lr": 1e-4,   "distributed": "ds",  "gpu_count": 4, "batch_size": 32,  "gradient_accumulation_steps": 1, "use_lora": True},
    "15_40_b":{"lr": 8e-5,   "distributed": "ds",  "gpu_count": 4, "batch_size": 16,  "gradient_accumulation_steps": 2, "use_lora": True},
    "40_80_b":{"lr": 8e-5,   "distributed": "ds",  "gpu_count": 8, "batch_size": 16,  "gradient_accumulation_steps": 2, "use_lora": True},
}

for _key in SFT_ENV_SIZE_CONFIG:
    SFT_ENV_SIZE_CONFIG[_key]["label"] = _key


def get_sft_env_config(param_nums: int) -> dict:
    if param_nums is None:
        raise ValueError("Cannot determine model size: weight counting failed")
    result = {"lr": 4e-5, "distributed": "ds", "gpu_count": 8, "batch_size": 6, "use_lora": True}
    if param_nums < 1_000_000_000:
        result = SFT_ENV_SIZE_CONFIG["0_1_b"]
    elif param_nums < 2_000_000_000:
        result = SFT_ENV_SIZE_CONFIG["1_2_b"]
    elif param_nums < 4_000_000_000:
        result = SFT_ENV_SIZE_CONFIG["2_4_b"]
    elif param_nums < 5_000_000_000:
        result = SFT_ENV_SIZE_CONFIG["4_5_b"]
    elif param_nums < 9_000_000_000:
        result = SFT_ENV_SIZE_CONFIG["5_9_b"]
    elif param_nums < 12_000_000_000:
        result = SFT_ENV_SIZE_CONFIG["9_12_b"]
    elif param_nums < 15_000_000_000:
        result = SFT_ENV_SIZE_CONFIG["12_15_b"]
    elif param_nums < 40_000_000_000:
        result = SFT_ENV_SIZE_CONFIG["15_40_b"]
    elif param_nums < 80_000_000_000:
        result = SFT_ENV_SIZE_CONFIG["40_80_b"]
    else:
        _log(f"Model size {param_nums} is not supported, using 40_80_b")
    return deepcopy(result)


def get_generation_time_ratio(param_nums: int) -> float:
    """Fraction of total time budget spent on data generation, by model size.

    Smaller models train faster, so they can afford to spend more of the
    fixed wall-clock budget generating data.
    """
    if param_nums < 2_000_000_000:
        return 0.25
    elif param_nums < 4_000_000_000:
        return 0.2
    elif param_nums < 6_000_000_000:
        return 0.15
    else:
        return 0.12


def get_run_cmd(config: dict, gpu_nums: int) -> str:
    required_keys = [
        "epoch_num",
        "batch_size",
        "learning_rate",
        "min_lr_rate",
        "use_liger_kernel",
        "optimizer",
        "use_lora",
        "packing",
        "disable_fa",
    ]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Required key {key} not found in config")

    gpu_nums = get_gpu_count()
    run_type = config["distributed"]
    if gpu_nums > 1 and run_type == "ddp":
        start_cmd = f"torchrun --nproc_per_node={gpu_nums}"
    elif run_type == "ds":
        start_cmd = "deepspeed"
    else:
        start_cmd = "python"

    template = (
        start_cmd
        + """ train_sft_env.py \
    --request_path {request_path} \
    --bf16 True \
    --report_to wandb \
    --output_dir {output_dir} \
    --num_train_epochs {epoch_num} \
    --per_device_train_batch_size {batch_size} \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps {gradient_accumulation_steps} \
    --eval_accumulation_steps 1 \
    --eval_strategy no \
    --save_strategy epoch \
    --logging_steps 5 \
    --learning_rate {learning_rate} \
    --weight_decay 0. \
    --warmup_steps 35 \
    --lr_scheduler_type cosine_with_min_lr \
    --lr_scheduler_kwargs "{\\"min_lr_rate\\": {min_lr_rate}}" \
    --tf32 True \
    --gradient_checkpointing {gradient_checkpointing} \
    --optim {optimizer} \
    --use_liger_kernel {use_liger_kernel} \
    --packing {packing} \
    --disable_fa {disable_fa} \
    --max_length 4096"""
    )

    if run_type == "ds":
        template += " --deepspeed ds_config/zero3.json"

    if config.get("use_lora", False):
        template += " --use_lora True"

    if not config.get("disable_fa", False):
        template += " --padding_free True"

    for key, value in config.items():
        template = template.replace("{" + key + "}", str(value))

    return template


def get_training_json(train_info: dict) -> dict:
    model_path = train_info["model_path"]
    model_architecture = get_model_architecture(model_path)
    param_nums = get_model_num_params(model_path)
    config = get_sft_env_config(param_nums)

    task_id = train_info["task_id"]
    env_names = train_info.get("dataset_type", {}).get("environment_names") or ["liars_dice"]
    dataset_path = f"/workspace/scripts/datasets/sft_env_{task_id}"

    run_config = {
        "epoch_num": 1,
        "batch_size": config["batch_size"],
        "learning_rate": config["lr"],
        "min_lr_rate": 0.25,
        "use_liger_kernel": get_use_liger(model_architecture),
        "optimizer": "paged_adamw_8bit",
        "use_lora": config.get("use_lora", False),
        "disable_fa": disable_flash_attention(model_architecture),
        "packing": "False",  # pre-tokenised dataset; TRL packing not used
        "gpu_nums": config["gpu_count"],
        "output_dir": train_info["output_dir"],
        "request_path": train_info["request_path"],
        "distributed": config.get("distributed", "ddp"),
        "gradient_checkpointing": "True",
        "gradient_accumulation_steps": config["gradient_accumulation_steps"],
    }

    if train_info.get("find_lk_lr"):
        lr = get_lr_from_ar_instruct(model_architecture, param_nums)
        if lr is not None:
            _log(f"Using lr from architecture config: {lr}", flush=True)
            run_config["learning_rate"] = lr
        else:
            _log(f"Using lr from config: {run_config['learning_rate']}", flush=True)

    run_config["learning_rate"] *= train_info["reg_ratio"]

    run_cmd = get_run_cmd(run_config, run_config["gpu_nums"])

    train_request = deepcopy(train_info)
    train_request["dataset_path"] = dataset_path
    train_request["save_before_remaining_time"] = 5
    train_request["adjust_batch_size"] = False
    train_request["periodic_save_steps"] = 200
    train_request["checking_step"] = 70
    train_request["min_steps"] = max(
        int(train_info["hours_to_complete"] * 70),
        train_info.get("min_steps", 100),
    )

    gen_seconds = int(train_info["hours_to_complete"] * 3600 * get_generation_time_ratio(param_nums))
    generate_cmd = (
        f"python -m envs.generate_trajectories"
        f" --environment_names {' '.join(env_names)}"
        f" --output_path {dataset_path}"
        f" --time_limit_seconds {gen_seconds}"
    )

    print("Run command:", run_cmd)

    return {
        "train_request": train_request,
        "run_cmd": run_cmd,
        "generate_cmd": generate_cmd,
    }
