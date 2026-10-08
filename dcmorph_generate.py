import torch
import time
import matplotlib.pyplot as plt
import cv2
import numpy as np
import logging
import os
import torchvision.transforms as T

from pathlib import Path
from PIL import Image
from insightface.app import FaceAnalysis
from diffusers import StableDiffusionPipeline, DDIMScheduler, AutoencoderKL 
from ip_adapter.custom_pipelines import StableDiffusionXLCustomPipeline, StableDiffusionXLCustomPipelineMorph
from ip_adapter.ip_adapter_faceid import IPAdapterFaceID, IPAdapterFaceIDXL

def clear_cache(img):
    del img
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


def detect_face(img_name, imgs_path):
    img_path = imgs_path + img_name
    image = cv2.imread(img_path)
    faces = app.get(image)
    assert len(faces) > 0
    faceid_embeds = torch.from_numpy(faces[0].normed_embedding).unsqueeze(0)
    return faceid_embeds


def save_image(image, name1, name2, output_dir, ext=".png"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{name1}-vs-{name2}{ext}"
    output_path = output_dir / filename

    image.save(output_path)


def align_face(image: np.ndarray, kps: np.ndarray, output_size=(112, 112)) -> np.ndarray:
    # Standard reference points (112x112) used in InsightFace
    ref_pts = np.array([
        [38.2946, 51.6963],   # left eye
        [73.5318, 51.5014],   # right eye
        [56.0252, 71.7366],   # nose
        [41.5493, 92.3655],   # left mouth
        [70.7299, 92.2041],   # right mouth
    ], dtype=np.float32)

    kps = np.array(kps, dtype=np.float32)

    # Compute affine transform
    M = cv2.estimateAffinePartial2D(kps, ref_pts, method=cv2.LMEDS)[0]

    # Warp image
    aligned = cv2.warpAffine(image, M, output_size, borderValue=0.0)
    return aligned


def get_image_batch(img1_name, img2_name, imgs_path):
    transform = T.Compose([
        T.Resize((h,w), interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop((h, w)),
        T.ToTensor(),                 # [0,1]
        T.Normalize([0.5]*3, [0.5]*3) # [-1,1]
    ])

    img_path_1 = imgs_path + img1_name
    image_1_pil = Image.open(img_path_1).convert("RGB")
    image_1_pil.resize((h, w))
    image_1_pil = transform(image_1_pil).unsqueeze(0)

    img_path_2 = imgs_path + img2_name
    image_2_pil = Image.open(img_path_2).convert("RGB")
    image_2_pil.resize((h, w))
    image_2_pil = transform(image_2_pil).unsqueeze(0)

    return torch.cat([image_1_pil, image_2_pil], dim=0)


if __name__ == "__main__":
    base_model_path = "SG161222/RealVisXL_V3.0"
    vae_model_path = "stabilityai/sd-vae-ft-mse"


    pairs = "TODO" # text file for pairs to morph
    pairs_txt = Path(pairs_male_smiling)
    image_path = "TODO" # images path
    ip_ckpt ="./ip-adapter-faceid_sdxl.bin" # model path

    output_path = "./generated_dcmorph"
    output_path_processed = "./generated_processed_dcmoprh"

    device = "cuda"
    debug = False
    prompt = "headshot, high quality, best quality" #  "headshot, high quality, best quality"
    negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality, blurry"
    guidance_scale = 2
    w = 1024
    h = 1024
    alpha = 0.5

    app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(224, 224))
    rec_model = app.models["recognition"]


    noise_scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
        steps_offset=1,
    )

    pipe = StableDiffusionXLCustomPipelineMorph.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        scheduler=noise_scheduler,
        add_watermarker=False,
    ).to(device)

    vae = AutoencoderKL.from_pretrained(vae_model_path).to(dtype=torch.float16).to(device)

    # load ip-adapter
    ip_model = IPAdapterFaceIDXL(pipe, ip_ckpt, device)

    # Loop
    fldr_name = pairs_txt.stem
    num_lines = sum(1 for _ in pairs_txt.open("r"))
    start_time = time.time()  # start timer
    output_path_pair = os.path.join(output_path, fldr_name)
    output_path_processed_pair = os.path.join(output_path_processed, fldr_name)

    print(f"Start executing <<{fldr_name}>> of length {num_lines}")
    with pairs_txt.open("r") as f:
        for line in f:
            line = line.strip()
            img1_name, img2_name = line.split()

            # get face embeddings
            faceid_embeds_1 = detect_face(img1_name, imgs_path=image_path)
            faceid_embeds_2 = detect_face(img2_name, imgs_path=image_path)
            image_batch = get_image_batch(img1_name, img2_name, imgs_path=image_path)
    
            # generate image
            with torch.no_grad():
                image = ip_model.morph_inv(
                    prompt=prompt, negative_prompt=negative_prompt, faceid_embeds_1=faceid_embeds_1, faceid_embeds_2=faceid_embeds_2,  
                    num_samples=1, width=w, height=h, num_inference_steps=30, guidance_scale=guidance_scale, seed=2023, 
                    morph_sph_inter=False, image_inversion=True, start_image=image_batch, start_ratio=0.0, scale=alpha
                ) 
            
            if debug:
                plt.figure(figsize=(8, 4))
                plt.imshow(image_grid(image, 1, 1))
                plt.axis('off') # hide axes
                plt.show()
            save_image(image[0],  Path(img1_name).stem,  Path(img2_name).stem, output_dir=output_path_pair)
    
            # Align image
            image_np = np.array(image[0])
            image_pp = app.get(image_np)
            assert len(image_pp) > 0
            aligned_face = align_face(image_np, image_pp[0].kps) 
            save_image(Image.fromarray(aligned_face),  Path(img1_name).stem,  Path(img2_name).stem, output_dir=output_path_processed_pair)
    
            # clear cache
            clear_cache(image)
        
        end_time = time.time()  # end timer
        print(f"Total loop execution time: {end_time - start_time:.2f} seconds")

