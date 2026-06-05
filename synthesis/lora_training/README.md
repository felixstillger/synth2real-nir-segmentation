# LoRA Training for NIR Image Synthesis

This directory contains the necessary information to reproduce the LoRA fine-tuning process for the SDXL model used to synthesize NIR images from GTA5 RGB images.

## Setup

The training was performed using the standard `train_text_to_image_lora_sdxl.py` reference script from the HuggingFace `diffusers` library.

## Training Command

The following exact command was used for training:

```bash
accelerate launch train_text_to_image_lora_sdxl.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \
  --pretrained_vae_model_name_or_path="madebyollin/sdxl-vae-fp16-fix" \
  --train_data_dir="/path/to/ranus_lora" \
  --output_dir="sdxl_lora_base_rank8_no_text_encoder" \
  --resolution=1024 \
  --train_batch_size=2 \
  --gradient_accumulation_steps=1 \
  --max_train_steps=2000 \
  --learning_rate=0.0001 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --rank=8 \
  --checkpointing_steps=50 \
  --validation_prompt="nir, urban street intersection, driver view, car, bus, person, road, sidewalk, tree, traffic light, building" \
  --validation_epochs=1 \
  --mixed_precision="no" \
  --seed=0
```

## Key Parameters

* **LoRA Rank**: 8
* **Learning Rate**: 1e-4 (constant)
* **Text Encoder**: Not trained (frozen)
* **Iterations**: 2000 steps
* **Dataset**: 64 images selected from the RANUS NIR training split. The filenames are listed in `training_images.txt`.
* **Captions**: Captions were generated using Gemini 2.5 Flash, structured using the `gemini_captioning.py` script, and flattened with a `<nirstyle>` or `nir,` prefix. See `metadata.jsonl` for the exact caption assigned to each training image.

## Final Weights

The final weights used for inference were taken from `checkpoint-2000/pytorch_lora_weights.safetensors`.
