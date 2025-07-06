#!/usr/bin/env python3
"""
Memory optimization patch for finetune-freeze.py

This script adds memory optimization without modifying the TrainingConfig class.
Run this script before your training command.
"""

import os
import sys

# Set PyTorch memory optimization environment variable
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
print("✅ Set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")

# Add the memory optimization patch to the main script
filepath = "finetune-freeze.py"
with open(filepath, "r") as f:
    content = f.read()

# Add gradient checkpointing if not already present
if "model.gradient_checkpointing_enable()" not in content:
    # Find the line after model loading
    target_line = "model = EsmForMaskedLM.from_pretrained(config.base_model, config=model_config)"
    replacement = target_line + "\n    # Memory optimization: Enable gradient checkpointing\n    model.gradient_checkpointing_enable()\n    print(\"✅ Gradient checkpointing enabled for memory optimization\")"
    
    # Replace the target line
    content = content.replace(target_line, replacement)
    print("✅ Added gradient checkpointing")

# Modify the ClearEvalMemoryTrainer to clear memory more aggressively
if "def training_step(self, *args: Any, **kwargs: Any) -> float:" not in content:
    # Find the class definition
    target_class = "class ClearEvalMemoryTrainer(Trainer):"
    training_step_code = """    def training_step(self, *args: Any, **kwargs: Any) -> float:
        \"\"\"Run training step and periodically clear cache.\"\"\"
        loss = super().training_step(*args, **kwargs)
        # Clear cache every 50 steps to prevent gradual memory buildup
        if self.state.global_step % 50 == 0:
            self._clear_cuda_cache()
        return loss
"""
    # Find position to insert
    class_pos = content.find(target_class)
    evaluate_method_pos = content.find("    def evaluate(self", class_pos)
    
    # Insert the training_step method before evaluate
    new_content = content[:evaluate_method_pos] + training_step_code + content[evaluate_method_pos:]
    content = new_content
    print("✅ Enhanced memory clearing in ClearEvalMemoryTrainer")

# Write modified content back to file
with open(filepath, "w") as f:
    f.write(content)

print("\n📝 Memory optimization patches applied to finetune-freeze.py")
print("\n💡 Next steps:")
print("1. Use the new optimized-config.yml configuration file")
print("2. Run your training command: python finetune-freeze.py --config optimized-config.yml")
print("\n⚠️  Warning: These changes modify your original script. Make a backup if needed.")