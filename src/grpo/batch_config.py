"""Batch-size validation for GRPO group-size experiments."""


def validate_group_batch(num_generations, per_device_batch_size, gradient_accumulation_steps, world_size=1):
    """Return effective batch size; ensure complete groups per optimizer step."""
    if num_generations not in (2, 4, 8):
        raise ValueError("--num-generations must be one of 2, 4, 8")
    if min(per_device_batch_size, gradient_accumulation_steps, world_size) < 1:
        raise ValueError("Batch size, accumulation and world size must be positive")
    effective = per_device_batch_size * gradient_accumulation_steps * world_size
    if effective % num_generations:
        raise ValueError(
            f"Effective batch size {effective} must be divisible by "
            f"num_generations={num_generations}"
        )
    return effective
