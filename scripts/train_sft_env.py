"""
SFT trainer for EnvTask imitation learning.

Baseline: train_instruct.py (production machinery: callbacks, batch-size adjustment,
  success.txt, LoRA helpers).
Differences from train_instruct.py:
  1. Dataset: loaded via load_from_disk (HF DatasetDict) instead of MyDataset.
  2. Trainer: SFTTrainer with tokenize_and_mask (assistant-only loss) instead of Trainer.
"""

import datetime
import json
import os
from dataclasses import dataclass, field
from typing import Optional

import bitsandbytes as bnb
import torch
import transformers
import datasets as hf_datasets
from datasets import DatasetDict, load_from_disk
from peft import LoraConfig, TaskType as PeftTaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoTokenizer, BitsAndBytesConfig
from transformers.trainer_utils import is_main_process
from trl import SFTConfig, SFTTrainer

from customized_trainer import (
    CustomEvalSaveCallback,
    WhenToEvalHandler,
    resize_if_needed,
    set_generation_config,
)
from state_manager import get_state, set_state
from utility import log_info

LOCAL_RANK = int(os.getenv("LOCAL_RANK", "0"))


# ---------------------------------------------------------------------------
# Argument dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SFTEnvTrainingArgs(SFTConfig):
    request_path: Optional[str] = field(default=None)
    use_lora: Optional[bool] = field(default=False)
    disable_fa: Optional[bool] = field(default=False)


@dataclass
class LoraArguments:
    lora_r: int = 256
    lora_alpha: int = 512
    lora_dropout: float = 0.0
    lora_target_modules: str = "all"
    lora_weight_path: str = ""
    lora_bias: str = "none"
    q_lora: bool = False


# ---------------------------------------------------------------------------
# Model helpers (mirrored from train_instruct.py)
# ---------------------------------------------------------------------------

def find_all_linear_names(model):
    names = set()
    for name, module in model.named_modules():
        if isinstance(module, (bnb.nn.Linear4bit, torch.nn.Linear)):
            parts = name.split(".")
            names.add(parts[0] if len(parts) == 1 else parts[-1])
    names.discard("lm_head")
    return list(names)


def load_lora_model(training_args: SFTEnvTrainingArgs, model_path: str,
                    lora_args: LoraArguments, token_nums: int):
    if training_args.use_liger_kernel:
        from liger_kernel.transformers import AutoLigerKernelForCausalLM
        model_class = AutoLigerKernelForCausalLM
    else:
        model_class = transformers.AutoModelForCausalLM

    model = model_class.from_pretrained(
        model_path,
        attn_implementation="flash_attention_2" if not training_args.disable_fa else "eager",
        torch_dtype=torch.bfloat16,
        quantization_config=(
            BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                attn_implementation="flash_attention_2" if not training_args.disable_fa else "eager",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            if lora_args.q_lora
            else None
        ),
    )

    if lora_args.lora_target_modules == "all":
        target_modules = find_all_linear_names(model)
    else:
        target_modules = [m.strip() for m in lora_args.lora_target_modules.split() if m.strip()]

    lora_config = LoraConfig(
        r=lora_args.lora_r,
        lora_alpha=lora_args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_args.lora_dropout,
        bias=lora_args.lora_bias,
        task_type=PeftTaskType.CAUSAL_LM,
    )

    if lora_args.q_lora:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=training_args.gradient_checkpointing
        )

    model = get_peft_model(model, lora_config)

    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()

    model.config.use_cache = False
    if hasattr(model.config, "output_router_logits"):
        setattr(model.config, "output_router_logits", True)

    return model


def load_model(training_args: SFTEnvTrainingArgs, model_path: str, token_nums: int):
    model_class = transformers.AutoModelForCausalLM
    if training_args.use_liger_kernel:
        from liger_kernel.transformers import AutoLigerKernelForCausalLM
        log_info("Using LIGER kernel")
        model_class = AutoLigerKernelForCausalLM

    model = model_class.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2" if not training_args.disable_fa else "eager",
    )
    return model


# ---------------------------------------------------------------------------
# Tokenisation + masking (from game-trajectories-sft/train.py)
# ---------------------------------------------------------------------------

def tokenize_and_mask(dataset: DatasetDict, tokenizer, max_length: int = 4096) -> DatasetDict:
    """Apply chat template and mask non-assistant tokens so loss is assistant-only."""
    def _process(example):
        msgs = example["messages"]
        tools = example.get("tools") or None
        if isinstance(tools, str):
            # Tools are stored as a JSON string, not a list of dicts: Arrow unifies
            # struct schemas across a list column's elements, which corrupts
            # heterogeneous tool `parameters.properties` (e.g. a no-arg tool's `{}`
            # becomes `{"some_other_tool_arg": None}`). JSON-encoding avoids that.
            tools = json.loads(tools)
        # Same Arrow-struct-unification issue applies to tool_calls[].function.arguments
        # across envs (e.g. game envs' {"action_id": int} vs intercode's {"command": str}):
        # generators store `arguments` JSON-encoded as a string. Decode back to a dict
        # here so the chat template's `tojson` filter renders it correctly (it would
        # double-encode an already-JSON string).
        msgs = [
            {
                **m,
                "tool_calls": [
                    {
                        **call,
                        "function": {
                            **call["function"],
                            "arguments": (
                                json.loads(call["function"]["arguments"])
                                if isinstance(call["function"].get("arguments"), str)
                                else call["function"].get("arguments")
                            ),
                        },
                    }
                    for call in m["tool_calls"]
                ],
            }
            if m.get("tool_calls")
            else m
            for m in msgs
        ]
        ids = tokenizer.apply_chat_template(msgs, tools=tools, tokenize=True, add_generation_prompt=False)
        mask = [0] * len(ids)
        for i, msg in enumerate(msgs):
            if msg["role"] != "assistant":
                continue
            p = len(tokenizer.apply_chat_template(msgs[:i],   tools=tools, tokenize=True, add_generation_prompt=True))
            r = len(tokenizer.apply_chat_template(msgs[:i+1], tools=tools, tokenize=True, add_generation_prompt=False))
            for j in range(p, r):
                mask[j] = 1
        if len(ids) > max_length:
            ids, mask = ids[:max_length], mask[:max_length]
        return {"input_ids": ids, "assistant_masks": mask}

    log_info("Tokenizing + masking dataset...")
    hf_datasets.disable_progress_bar()
    result = dataset.map(_process, num_proc=4)
    hf_datasets.enable_progress_bar()
    log_info("Tokenizing + masking done.")
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    argument_parser = transformers.HfArgumentParser((SFTEnvTrainingArgs, LoraArguments))
    (training_args, lora_args) = argument_parser.parse_args_into_dataclasses()

    train_info = json.load(open(training_args.request_path, "r"))
    train_request = train_info["train_request"]
    task_id = train_request["task_id"]

    tokenizer = AutoTokenizer.from_pretrained(train_request["model_path"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load pre-generated trajectories and apply assistant masking
    raw: DatasetDict = load_from_disk(train_request["dataset_path"])
    log_info(f"Dataset: train={len(raw['train'])}")
    dataset = tokenize_and_mask(raw, tokenizer, max_length=training_args.max_length or 4096)

    train_ds = dataset["train"]
    log_info(f"train_size: {len(train_ds)}")

    # Batch size adjustment (mirrors train_instruct.py)
    original_steps = len(train_ds) // (
        training_args.per_device_train_batch_size
        * training_args.gradient_accumulation_steps
        * training_args.world_size
    )
    log_info(f"original_steps: {original_steps}")

    max_batch_size_theory = len(train_ds) / (
        training_args.gradient_accumulation_steps
        * training_args.world_size
        * train_request["min_steps"]
    )
    max_batch_size_theory = max(int(max_batch_size_theory), 1)

    if (training_args.per_device_train_batch_size > max_batch_size_theory
            and train_request.get("adjust_batch_size", True)):
        log_info(
            f"Reducing batch size {training_args.per_device_train_batch_size} → {max_batch_size_theory}"
        )
        training_args.per_device_train_batch_size = max_batch_size_theory

    # Load model
    if training_args.use_lora:
        model = load_lora_model(training_args, train_request["model_path"], lora_args, len(tokenizer))
    else:
        model = load_model(training_args, train_request["model_path"], len(tokenizer))
        resize_if_needed(train_request["model_name"], model, len(tokenizer))

    try:
        model.config.use_cache = False
    except Exception:
        pass

    set_generation_config(train_request["model_name"], model)

    if is_main_process(LOCAL_RANK):
        os.makedirs(training_args.output_dir, exist_ok=True)
        log_info(f"Output dir: {training_args.output_dir}")

    periodic_save_steps = train_request.get("periodic_save_steps", -1)
    max_steps = train_request.get("max_steps", -1)
    training_args.save_only_model = True

    total_steps_per_epoch = len(train_ds) // (
        training_args.per_device_train_batch_size
        * training_args.gradient_accumulation_steps
        * training_args.world_size
    )
    total_steps_all_epochs = total_steps_per_epoch * training_args.num_train_epochs
    log_info(
        f"total_steps_per_epoch: {total_steps_per_epoch}; "
        f"total_steps_all_epochs: {total_steps_all_epochs}"
    )

    start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    state = get_state()
    state["train"]["start_train_time"] = start_time
    if is_main_process(LOCAL_RANK):
        set_state(state)

    success_file = os.path.join(training_args.output_dir, "success.txt")
    if is_main_process(LOCAL_RANK) and os.path.exists(success_file):
        os.remove(success_file)

    checking_step = train_request.get("checking_step", 70)
    if checking_step >= total_steps_per_epoch:
        checking_step = max(total_steps_per_epoch - 2, 1)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        processing_class=tokenizer,
        callbacks=[
            CustomEvalSaveCallback(
                WhenToEvalHandler(
                    train_request["end_time"],
                    train_request["save_before_remaining_time"],
                    periodic_save_steps=periodic_save_steps,
                    steps_per_epoch=total_steps_per_epoch,
                    max_steps=max_steps,
                ),
                train_request["submission_dir"],
                training_args.output_dir,
                train_request["model_name"],
                max_steps,
                checking_step=checking_step,
                total_steps_all_epochs=total_steps_all_epochs,
                end_time=train_request["end_time"],
                checking_mode=train_request.get("checking_mode", "none"),
            )
        ],
    )

    trainer.train()

    if is_main_process(LOCAL_RANK):
        with open(success_file, "w") as f:
            f.write("Success")
    log_info("Training successfully done", "finish")


if __name__ == "__main__":
    main()
