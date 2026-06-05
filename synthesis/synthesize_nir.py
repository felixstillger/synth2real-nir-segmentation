# run_experiment.py
import os
import argparse
from pathlib import Path
import torch
import numpy as np
import cv2
import json
from datetime import datetime
from PIL import Image
from diffusers import (
    ControlNetUnionModel,
    AutoencoderKL,
    StableDiffusionXLControlNetUnionImg2ImgPipeline,
)
from diffusers.utils import load_image
from controlnet_aux import NormalBaeDetector, MidasDetector
from tqdm import tqdm

def make_canny_condition(image):
    # extract canny edges from segmentation mask images, example code from huggingface diffusers 
    image = np.array(image)
    image = cv2.Canny(image, 100, 200)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    image = Image.fromarray(image)

    return image


def prepare_image(image, width=1024, height=1024):
    """Resize and crop image to target dimensions."""
    w, h = image.size
    if w < h:
        new_w = width
        new_h = int(height * h / w)
    else:
        new_h = height
        new_w = int(width * w / h)
    image = image.resize((new_w, new_h), resample=Image.LANCZOS)
    left = (new_w - width) / 2
    top = (new_h - height) / 2
    right = (new_w + width) / 2
    bottom = (new_h + height) / 2
    return image.crop((left, top, right, bottom))


def prepare_mask(image, width=1024, height=1024):
    """Resize and crop mask to target dimensions."""
    w, h = image.size
    if w < h:
        new_w = width
        new_h = int(height * h / w)
    else:
        new_h = height
        new_w = int(width * w / h)
    image = image.resize((new_w, new_h), resample=Image.NEAREST)
    left = (new_w - width) / 2
    top = (new_h - height) / 2
    right = (new_w + width) / 2
    bottom = (new_h + height) / 2
    return image.crop((left, top, right, bottom))


def make_linear_scheduler(start, end, decay_steps, total_steps):
    """Create linear scheduler for LoRA weights."""
    scales = torch.linspace(start, end, decay_steps).tolist()
    scales += [end] * (total_steps - decay_steps + 1)
    return scales


def process_batch_controls(image_paths, args, processor_midas, normal_bae, device):
    """Prepare control images for a batch."""
    batch_images = []
    batch_masks = []
    batch_canny = []
    batch_depth = []
    batch_normal = []
    valid_paths = []
    
    for img_path in image_paths:
        if not os.path.exists(img_path):
            print(f"Warning: Image {img_path} not found, skipping...")
            continue
            
        mask_path = str(img_path).replace("images", "labels")
        if not os.path.exists(mask_path):
            print(f"Warning: Mask for {img_path} not found, skipping...")
            continue
        
        # Load and prepare images
        image = prepare_image(load_image(str(img_path)))
        mask = prepare_mask(load_image(mask_path))
        canny_image = make_canny_condition(mask)
        depth_map_midas = processor_midas(image, detect_resolution=1024, image_resolution=1024, output_type='cv2')
        depth_map_midas = Image.fromarray(depth_map_midas)
        normal_map = normal_bae(image, hand_and_face=False, output_type='cv2')
        normal_map = cv2.resize(normal_map, (1024, 1024))
        normal_map = Image.fromarray(normal_map)
        
        batch_images.append(image)
        batch_masks.append(mask)
        batch_canny.append(canny_image)
        batch_depth.append(depth_map_midas)
        batch_normal.append(normal_map)
        valid_paths.append(img_path)
    
    return batch_images, batch_masks, batch_canny, batch_depth, batch_normal, valid_paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=str, help="Path to the source directory", default="../data/GTA5_prepared/images/train")
    parser.add_argument("--target-dir", type=str, help="Path to the target directory", default="../data/GTA5_prepared/nir_synth/train")
    parser.add_argument("--base-prompt", type=str, help="Base prompt for generation", default="nir, urban street, driver-view")
    parser.add_argument("--negative-prompt", type=str or None, help="Negative prompt for generation", default="(octane render, render, drawing, anime, bad photo, bad photography, graffiti, painting), (worst quality, low quality, blurry), (daylight colors, natural colors, warm tones, cool tones, color temperature, hue shift, color balance)") # default="(octane render, render, drawing, anime, bad photo, bad photography, graffiti, painting), (worst quality, low quality, blurry)"
    parser.add_argument("--max-images", type=int, help="Maximum number of images to process", default=None)
    parser.add_argument("--start-index", type=int, help="Starting index for image processing (default: 0)", default=0)
    parser.add_argument("--batch-size", type=int, help="Batch size for inference", default=4)
    args = parser.parse_args()

    # Initialize WandB
    # Manual run with selected parameters
    base_prompt = args.base_prompt

    negative_prompt = args.negative_prompt if args.negative_prompt not in [None, "", "none", "None", "NONE"] else None
    run_config={
            "scheduler_type": "scheduled",
            "strength": 0.99, # changed from 0.95 to 0.99 for stronger NIR spectrum adherence
            "num_inference_steps": 50,
            "scheduler_start": 1.2,
            "scheduler_end": 0.3,
            "scheduler_decay_steps": 30,
            "control_mode":  [1,3,4],
            "control_strength_canny": 0.9,
            "control_strength_depth": 0.2,
            "control_strength_normal": 0.3,
            "controlnet_version": "promax",
            "batch_size": args.batch_size,
            "base_prompt": base_prompt,
            "negative_prompt": negative_prompt if negative_prompt is not None else "",
        }
    
    cfg = argparse.Namespace(**run_config)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model components
    print("Loading models...")
    model_repo = "stabilityai/stable-diffusion-xl-base-1.0"
    if cfg.controlnet_version == "promax":
        controlnet = ControlNetUnionModel.from_pretrained("./controlnet-union-sdxl-1.0-promax", # requires local download and loading
                                                        torch_dtype=torch.float16,
                                                        use_safetensors=True,
        )
    else:
        controlnet = ControlNetUnionModel.from_pretrained(
            "xinsir/controlnet-union-sdxl-1.0", torch_dtype=torch.float16
        )
    vae = AutoencoderKL.from_pretrained(
        "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16
    )
    lora_checkpoint = "./lora_training/sdxl_lora_base_rank8_no_text_encoder/checkpoint-2000"
    processor_midas = MidasDetector.from_pretrained("lllyasviel/Annotators")
    normal_bae = NormalBaeDetector.from_pretrained("lllyasviel/Annotators").to(device)

    pipe = StableDiffusionXLControlNetUnionImg2ImgPipeline.from_pretrained(
        model_repo,
        controlnet=controlnet,
        vae=vae,
        torch_dtype=torch.float16,
        use_safetensors=True,
    ).to(device)
    pipe.load_lora_weights(lora_checkpoint, adapter_name="nir_lora")

     # Setup LoRA scheduling based on type
    if cfg.scheduler_type == "static":
        pipe.set_adapters("nir_lora", cfg.lora_scale)
        callback = None
        
    elif cfg.scheduler_type == "scheduled":
        total_steps = cfg.num_inference_steps
        lora_scales = make_linear_scheduler(
            cfg.scheduler_start, 
            cfg.scheduler_end, 
            cfg.scheduler_decay_steps, 
            total_steps
        )
        pipe.set_adapters("nir_lora", lora_scales[0])

        def callback(pipeline, step: int, timestep: torch.LongTensor, callback_kwargs: dict):
            pipeline.set_adapters("nir_lora", lora_scales[step + 1])
            return callback_kwargs
    else:
        raise ValueError(f"Unknown scheduler_type: {cfg.scheduler_type}")

    # GTA image folder
    all_image_paths = sorted(list(Path(args.source_dir).rglob("*.png")))
    target_dir = Path(args.target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    # Save run configuration to JSON file
    config_save_path = target_dir / "run_config.json"
    config_to_save = {
        "run_config": run_config,
        "arguments": {
            "source_dir": str(args.source_dir),
            "target_dir": str(args.target_dir),
            "base_prompt": args.base_prompt,
            "negative_prompt": args.negative_prompt,
            "max_images": args.max_images,
            "start_index": args.start_index,
            "batch_size": args.batch_size,
        },
        "timestamp": datetime.now().isoformat(),
        "device": str(device),
    }
    with open(config_save_path, 'w') as f:
        json.dump(config_to_save, f, indent=2)
    print(f"Run configuration saved to: {config_save_path}")

    # Apply start index and max images
    start_idx = args.start_index
    if start_idx >= len(all_image_paths):
        raise ValueError(f"Start index {start_idx} is beyond the total number of images ({len(all_image_paths)})")
    
    if args.max_images is not None:
        end_idx = min(start_idx + args.max_images, len(all_image_paths))
    else:
        end_idx = len(all_image_paths)
    
    image_paths = all_image_paths[start_idx:end_idx]
    
    print(f"\nDataset information:")
    print(f"  Total images available: {len(all_image_paths)}")
    print(f"  Start index: {start_idx}")
    print(f"  End index: {end_idx}")
    print(f"  Images to process: {len(image_paths)}")

    # Process in batches
    batch_size = args.batch_size
    num_batches = (len(image_paths) + batch_size - 1) // batch_size
    
    for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, len(image_paths))
        batch_paths = image_paths[batch_start:batch_end]
        
        print(f"\nProcessing batch {batch_idx+1}/{num_batches} ({len(batch_paths)} images)")
        
        # Prepare all control images for this batch
        batch_images, batch_masks, batch_canny, batch_depth, batch_normal, valid_paths = process_batch_controls(
            batch_paths, args, processor_midas, normal_bae, device
        )
        
        if len(batch_images) == 0:
            continue
        
        # Prepare batch inputs for pipeline
        # Need to duplicate prompts for batch size
        prompts = [base_prompt] * len(batch_images)
        neg_prompts = [negative_prompt] * len(batch_images) if negative_prompt else None
        
        # Stack control images - pipeline expects list of control image batches
        control_images_batch = [
            batch_depth,  # All depth maps
            batch_canny,  # All canny edges
            batch_normal  # All normal maps
        ]
        
        # Generate results for batch
        generator = torch.Generator(device=device).manual_seed(42)
        results = pipe(
            prompts,
            negative_prompt=neg_prompts,
            image=batch_images,
            control_image=control_images_batch,
            strength=float(run_config["strength"]),
            num_inference_steps=int(run_config["num_inference_steps"]),
            controlnet_conditioning_scale=[
                float(run_config["control_strength_depth"]), 
                float(run_config["control_strength_canny"]), 
                float(run_config["control_strength_normal"])
            ],
            generator=generator,
            control_mode=run_config["control_mode"],
            callback_on_step_end=callback,
        ).images
        
        # Save and log results
        for idx, (result, img_path, image, mask, canny, depth, normal) in enumerate(
            zip(results, valid_paths, batch_images, batch_masks, batch_canny, batch_depth, batch_normal)
        ):
            target_path = target_dir / img_path.name
            result.save(target_path)
            
        # Free up memory after each batch
        del batch_images, batch_masks, batch_canny, batch_depth, batch_normal, results
        torch.cuda.empty_cache()

    print("Experiment complete!")


if __name__ == "__main__":
    main()