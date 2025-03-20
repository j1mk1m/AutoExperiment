from share import *
import config
import argparse, os
from PIL import Image

import cv2
import einops
import numpy as np
import torch
import random
from einops import repeat

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.openpose import OpenposeDetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
from utils import get_folder_names_and_indexes


apply_openpose = OpenposeDetector()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = create_model('./models/controlnet/cldm_v15.yaml').cpu()
model.load_state_dict(load_state_dict('./models/controlnet/control_sd15_openpose.pth', location=device))
model = model.to(device)
ddim_sampler = DDIMSampler(model)


def process(input_image, prompt, n_prompt, num_samples, image_resolution, detect_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, sample_path):
    with torch.no_grad():
        input_image = HWC3(input_image)
        detected_map, _ = apply_openpose(resize_image(input_image, detect_resolution))
        detected_map = HWC3(detected_map)
        img = resize_image(input_image, image_resolution)
        H, W, C = img.shape

        detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_NEAREST)
        
        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)
        
        control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)
        
        c = repeat(prompt, 'b num_words embedding_size -> (repeat b) num_words embedding_size', repeat=num_samples)
        
        cond = {"c_concat": [control], "c_crossattn": [c]}
        un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
        shape = (4, H // 8, W // 8)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

        model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
        samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                     shape, cond, verbose=False, eta=eta,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=un_cond)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
        
        # Save the images
        print("Saving the images...")
        
        base_count = len(os.listdir(sample_path))
        for i, x_sample in enumerate(x_samples):
            img = Image.fromarray(x_sample.astype(np.uint8))
            img.save(os.path.join(sample_path, f"{base_count:05}.png"))
            base_count += 1
        
        results = [x_samples[i] for i in range(num_samples)]
        
        img = Image.fromarray(detected_map.astype(np.uint8))
        img.save(os.path.join(sample_path, "detected_map.png"))
    return [detected_map] + results


def main():
    parser = argparse.ArgumentParser()

    # Arguments ControlNet Human Pose
    parser.add_argument(
        "--input_image",
        type=str,
        nargs="?",
        default="./test_imgs/pose1.png",
        help="the path to the input image"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="chilled man on the beach",
        help="the prompt to render"
    )
    parser.add_argument(
        "--n_prompt",
        type=str,
        nargs="?",
        default="old, naked person, black and white picture, highly edited photo",
        help="what you do not want in the image"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=3,
        help="number of generated samples",
    )
    
    parser.add_argument(
        "--strength",
        type=float,
        default=1.0,
        help="control strenght: value between 0.0 and 2.0",
    )
    
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=20,
        help="number of ddim sampling steps (from 0 to 100)",
    )
    
    parser.add_argument(
        "--image_resolution",
        type=int,
        default=512,
        help="Image resolution (from 256 to 768)",
    )
    
    parser.add_argument(
        "--detect_resolution",
        type=int,
        default=512,
        help="OpenPose resolution (from 128 to 1024)",
    )
    
    parser.add_argument(
        "--guess_mode",
        action='store_true',
        help="Guess Mode",
    )
    
    parser.add_argument(
        "--scale",
        type=float,
        default=9.0,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty)) (from 0.1 to 30.0)",
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    
    parser.add_argument(
        "--eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling)",
    )
    
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="results/random"
    )
    
    

    # Added Arguments (for ITI-GEN)
    parser.add_argument(
        '--attr-list',
        type=str,
        default='Male,Skin_Tone,Age',
        help='input the attributes that you want to debias, separated by commas. Eg, Male,Eyeglasses,...'

    )
    parser.add_argument(
        '--prompt-path', 
        type=str,
        default='', 
        help='checkpoint of the learned token embeddings that are used for image generation in Stable Diffusion'
    )
    
    
    opt = parser.parse_args()
    input_image = cv2.imread(opt.input_image)
    input_image = np.array(input_image)

    os.makedirs(opt.outdir, exist_ok=True)
    emb = torch.load(opt.prompt_path).to(device)
    
    # Get combination
    folder_with_indexes = get_folder_names_and_indexes(opt.attr_list.split(','))

    for folder, index in folder_with_indexes.items():
        sample_path = os.path.join(opt.outdir, folder)
        os.makedirs(sample_path, exist_ok=True)
        
        prompt = emb[index].unsqueeze(0)
        
        process(input_image, prompt, opt.n_prompt, opt.num_samples,
                opt.image_resolution, opt.detect_resolution, opt.ddim_steps, opt.guess_mode,
                opt.strength, opt.scale, opt.seed, opt.eta, sample_path)

if __name__ == "__main__":
    main()