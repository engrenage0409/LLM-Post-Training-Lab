"""Construct a stage-two starting model from a base checkpoint and SFT LoRA."""

from pathlib import Path

from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM


def load_sft_merged_model(base_model: str, sft_adapter: str):
    """Merge the frozen stage-one adapter in memory; stage two adds a new LoRA.

    The merged SFT weights become the DPO reference when the new LoRA is
    disabled, and the GRPO initialization. Original files are unchanged.
    """
    adapter_path = Path(sft_adapter)
    if not (adapter_path / "adapter_config.json").is_file():
        raise FileNotFoundError(
            f"No PEFT adapter_config.json at {adapter_path}. Train SFT first."
        )
    config = PeftConfig.from_pretrained(str(adapter_path))
    expected = config.base_model_name_or_path
    if expected and str(expected) != str(base_model):
        raise ValueError(
            f"SFT adapter uses base {expected!r}, but --model is {base_model!r}"
        )
    model = AutoModelForCausalLM.from_pretrained(base_model, dtype="auto")
    model = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=False)
    return model.merge_and_unload()
