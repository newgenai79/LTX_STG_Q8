import json
import os
import random
from datetime import datetime
from pathlib import Path
from diffusers.utils import logging

import imageio
import numpy as np
from safetensors import safe_open
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import T5EncoderModel, T5Tokenizer
from omegaconf import OmegaConf

from ltx_video.models.autoencoders.causal_video_autoencoder import (
    CausalVideoAutoencoder,
)
from ltx_video.models.transformers.symmetric_patchifier import SymmetricPatchifier
from ltx_video.pipelines.pipeline_ltx_video import LTXVideoPipeline
from ltx_video.schedulers.rf import RectifiedFlowScheduler
from ltx_video.utils.conditioning_method import ConditioningMethod

from q8_kernels.models.LTXVideo import LTXTransformer3DModel
from q8_kernels.graph.graph import make_dynamic_graphed_callable

def load_vae(vae_config, ckpt):
    vae = CausalVideoAutoencoder.from_config(vae_config)
    vae_state_dict = {
        key.replace("vae.", ""): value
        for key, value in ckpt.items()
        if key.startswith("vae.")
    }
    vae.load_state_dict(vae_state_dict)
    if torch.cuda.is_available():
        vae = vae.cuda()
    return vae.to(torch.bfloat16)

def load_transformer(unet_path="konakona/ltxvideo_q8", type="q8_kernels"):
    transformer = LTXTransformer3DModel.from_pretrained(unet_path)
    return transformer

def load_scheduler(scheduler_config):
    return RectifiedFlowScheduler.from_config(scheduler_config)

def load_image_to_tensor_with_resize_and_crop(
    image_path, target_height=512, target_width=768
):
    image = Image.open(image_path).convert("RGB")
    input_width, input_height = image.size
    aspect_ratio_target = target_width / target_height
    aspect_ratio_frame = input_width / input_height
    if aspect_ratio_frame > aspect_ratio_target:
        new_width = int(input_height * aspect_ratio_target)
        new_height = input_height
        x_start = (input_width - new_width) // 2
        y_start = 0
    else:
        new_width = input_width
        new_height = int(input_width / aspect_ratio_target)
        x_start = 0
        y_start = (input_height - new_height) // 2

    image = image.crop((x_start, y_start, x_start + new_width, y_start + new_height))
    image = image.resize((target_width, target_height))
    frame_tensor = torch.tensor(np.array(image)).permute(2, 0, 1).float()
    frame_tensor = (frame_tensor / 127.5) - 1.0
    return frame_tensor.unsqueeze(0).unsqueeze(2)

def calculate_padding(
    source_height: int, source_width: int, target_height: int, target_width: int
) -> tuple[int, int, int, int]:
    pad_height = target_height - source_height
    pad_width = target_width - source_width

    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left

    return (pad_left, pad_right, pad_top, pad_bottom)

def convert_prompt_to_filename(text: str, max_len: int = 20) -> str:
    clean_text = "".join(
        char.lower() for char in text if char.isalpha() or char.isspace()
    )

    words = clean_text.split()

    result = []
    current_length = 0

    for word in words:
        new_length = current_length + len(word)

        if new_length <= max_len:
            result.append(word)
            current_length += len(word)
        else:
            break

    return "-".join(result)

def get_unique_filename(
    base: str,
    ext: str,
    prompt: str,
    seed: int,
    resolution: tuple[int, int, int],
    dir: Path,
    endswith=None,
    index_range=1000,
    stg_mode: str = "stg-a",
    stg_scale: float = 1.0,
    stg_block_idx: list = [20],
    do_rescaling: bool = True,
) -> Path:
    mode = "CFG" if stg_scale == 0 else stg_mode.upper()
    if mode == "CFG":
        suffix = f"{mode}"
    else:
        suffix = f"{mode}_{stg_scale}_{stg_block_idx}"
    if do_rescaling:
        suffix = f"{suffix}_rescaled"
    base_filename = f"{base}_{convert_prompt_to_filename(prompt, max_len=30)}_{seed}_{resolution[0]}x{resolution[1]}x{resolution[2]}_{suffix}"
    for i in range(index_range):
        filename = dir / f"{base_filename}_{i}{endswith if endswith else ''}{ext}"
        if not os.path.exists(filename):
            return filename
    raise FileExistsError(
        f"Could not find a unique filename after {index_range} attempts."
    )

def seed_everething(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def main():
    config_path = "config.yaml"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    config = OmegaConf.load(config_path)

    logger = logging.get_logger(__name__)

    logger.warning(f"Running generation with configuration: {config}")

    seed_everething(config.seed)

    output_dir = (
        Path(config.output_path)
        if config.output_path
        else Path(f"outputs/{datetime.today().strftime('%Y-%m-%d')}")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    if config.input_image_path:
        media_items_prepad = load_image_to_tensor_with_resize_and_crop(
            config.input_image_path, config.height, config.width
        )
    else:
        media_items_prepad = None

    height = config.height if config.height else media_items_prepad.shape[-2]
    width = config.width if config.width else media_items_prepad.shape[-1]
    num_frames = config.num_frames

    height_padded = ((height - 1) // 32 + 1) * 32
    width_padded = ((width - 1) // 32 + 1) * 32
    num_frames_padded = ((num_frames - 2) // 8 + 1) * 8 + 1

    padding = calculate_padding(height, width, height_padded, width_padded)

    logger.warning(
        f"Padded dimensions: {height_padded}x{width_padded}x{num_frames_padded}"
    )

    if media_items_prepad is not None:
        media_items = F.pad(
            media_items_prepad, padding, mode="constant", value=-1
        )
    else:
        media_items = None

    ckpt_path_model = Path(config.ckpt_path + "/ltx-video-2b-v0.9.safetensors")
    ckpt = {}
    with safe_open(ckpt_path_model, framework="pt", device="cpu") as f:
        metadata = f.metadata()
        for k in f.keys():
            ckpt[k] = f.get_tensor(k)

    configs = json.loads(metadata["config"])
    vae_config = configs["vae"]
    scheduler_config = configs["scheduler"]

    vae = load_vae(vae_config, ckpt)
    transformer = load_transformer()
    scheduler = load_scheduler(scheduler_config)
    patchifier = SymmetricPatchifier(patch_size=1)
    text_encoder = T5EncoderModel.from_pretrained(
        "PixArt-alpha/PixArt-XL-2-1024-MS", subfolder="text_encoder", torch_dtype=torch.bfloat16
    )
    tokenizer = T5Tokenizer.from_pretrained(
        "PixArt-alpha/PixArt-XL-2-1024-MS", subfolder="tokenizer"
    )

    transformer = transformer.to(torch.bfloat16)
    for b in transformer.transformer_blocks:
        b.to(dtype=torch.float)

    for n, m in transformer.transformer_blocks.named_parameters():
        if "scale_shift_table" in n:
            m.data = m.data.to(torch.bfloat16)

    torch.cuda.synchronize()
    transformer.forward = make_dynamic_graphed_callable(transformer.forward)

    submodel_dict = {
        "transformer": transformer,
        "patchifier": patchifier,
        "text_encoder": text_encoder,
        "tokenizer": tokenizer,
        "scheduler": scheduler,
        "vae": vae,
    }

    pipeline = LTXVideoPipeline(**submodel_dict)

    sample = {
        "prompt": config.prompt,
        "prompt_attention_mask": None,
        "negative_prompt": config.negative_prompt,
        "negative_prompt_attention_mask": None,
        "media_items": media_items,
    }

    generator = torch.Generator(
        device="cuda" if torch.cuda.is_available() else "cpu"
    ).manual_seed(config.seed)

    images = pipeline(
        num_inference_steps=config.num_inference_steps,
        num_images_per_prompt=config.num_images_per_prompt,
        guidance_scale=config.guidance_scale,
        stg_mode=config.stg_mode,
        stg_scale=config.stg_scale,
        stg_block_idx=config.stg_block_idx,
        do_rescaling=config.do_rescaling,
        rescaling_scale=config.rescaling_scale,
        generator=generator,
        output_type="pt",
        height=height_padded,
        width=width_padded,
        num_frames=num_frames_padded,
        frame_rate=config.frame_rate,
        **sample,
        is_video=True,
        vae_per_channel_normalize=True,
        conditioning_method=(
            ConditioningMethod.FIRST_FRAME
            if media_items is not None
            else ConditioningMethod.UNCONDITIONAL
        ),
        image_cond_noise_scale=config.image_cond_noise_scale,
        mixed_precision=False,
        low_vram=True,
        transformer_type="q8_kernels"
    ).images

    (pad_left, pad_right, pad_top, pad_bottom) = padding
    pad_bottom = -pad_bottom
    pad_right = -pad_right
    if pad_bottom == 0:
        pad_bottom = images.shape[3]
    if pad_right == 0:
        pad_right = images.shape[4]
    images = images[:, :, :num_frames, pad_top:pad_bottom, pad_left:pad_right]

    for i in range(images.shape[0]):
        video_np = images[i].permute(1, 2, 3, 0).cpu().float().numpy()
        video_np = (video_np * 255).astype(np.uint8)
        fps = config.frame_rate
        height, width = video_np.shape[1:3]
        if video_np.shape[0] == 1:
            output_filename = get_unique_filename(
                f"image_output_{i}",
                ".png",
                prompt=config.prompt,
                seed=config.seed,
                resolution=(height, width, num_frames),
                dir=output_dir,
            )
            imageio.imwrite(output_filename, video_np[0])
        else:
            if config.input_image_path:
                base_filename = f"img_to_vid_{i}"
            else:
                base_filename = f"text_to_vid_{i}"
            output_filename = get_unique_filename(
                base_filename,
                ".mp4",
                prompt=config.prompt,
                seed=config.seed,
                resolution=(height, width, num_frames),
                dir=output_dir,
                stg_mode=config.stg_mode,
                stg_scale=config.stg_scale,
                stg_block_idx=config.stg_block_idx,
                do_rescaling=config.do_rescaling,
            )

            with imageio.get_writer(output_filename, fps=fps) as video:
                for frame in video_np:
                    video.append_data(frame)

            if config.input_image_path:
                reference_image = (
                    (
                        media_items_prepad[0, :, 0].permute(1, 2, 0).cpu().data.numpy()
                        + 1.0
                    )
                    / 2.0
                    * 255
                )
                imageio.imwrite(
                    get_unique_filename(
                        base_filename,
                        ".png",
                        prompt=config.prompt,
                        seed=config.seed,
                        resolution=(height, width, num_frames),
                        dir=output_dir,
                        endswith="_condition",
                    ),
                    reference_image.astype(np.uint8),
                )
        logger.warning(f"Output saved to {output_dir}")

if __name__ == "__main__":
    main()
