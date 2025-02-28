import os
from io import BytesIO

import torch
from torch import nn
from torch.nn import functional as F
import torchgeometry as tgm
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from PIL import Image
import numpy as np
from torchvision import transforms
import torchvision.models.detection as detection

from models import SegGenerator, GMM, ALIASGenerator
from utils import gen_noise, load_checkpoint, save_images

app = FastAPI()

# Static configuration (replacing argparse)
class Opt:
    name = 'virtual_tryon'
    load_height = 1024
    load_width = 768
    dataset_dir = './datasets/'
    checkpoint_dir = './checkpoints/'
    save_dir = './results/'
    semantic_nc = 13
    init_type = 'xavier'
    init_variance = 0.02
    grid_size = 5
    norm_G = 'spectralaliasinstance'
    ngf = 64
    num_upsampling_layers = 'most'

opt = Opt()

# Load models
seg = SegGenerator(opt, input_nc=opt.semantic_nc + 8, output_nc=opt.semantic_nc)
gmm = GMM(opt, inputA_nc=7, inputB_nc=3)
opt.semantic_nc = 7
alias = ALIASGenerator(opt, input_nc=9)
opt.semantic_nc = 13

load_checkpoint(seg, os.path.join(opt.checkpoint_dir, 'seg_final.pth'))
load_checkpoint(gmm, os.path.join(opt.checkpoint_dir, 'gmm_final.pth'))
load_checkpoint(alias, os.path.join(opt.checkpoint_dir, 'alias_final.pth'))

seg.eval()
gmm.eval()
alias.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((opt.load_width, opt.load_height), interpolation=2),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Placeholder for pose estimation and segmentation
pose_model = detection.keypointrcnn_resnet50_fpn(pretrained=True)
pose_model.eval()

def get_pose_keypoints(image):
    img_tensor = transforms.ToTensor()(image).unsqueeze(0)
    with torch.no_grad():
        predictions = pose_model(img_tensor)[0]
        keypoints = predictions['keypoints'][0].detach().numpy()[:, :2]  # Top person's keypoints
    return keypoints

def get_segmentation(image):
    # Placeholder: Replace with actual segmentation model
    parse = np.zeros((opt.load_height, opt.load_width), dtype=np.uint8)
    return Image.fromarray(parse)

def get_agnostic(image, parse, pose_data):
    # Simplified agnostic image generation
    agnostic = image.copy()
    # Add masking logic here if needed
    return agnostic

@app.post("/tryon/")
async def virtual_tryon(human_image: UploadFile = File(...), cloth_image: UploadFile = File(...)):
    # Read uploaded images
    human_img = Image.open(BytesIO(await human_image.read())).convert('RGB')
    cloth_img = Image.open(BytesIO(await cloth_image.read())).convert('RGB')

    # Preprocess images
    human_img_resized = transform(human_img).unsqueeze(0)
    cloth_img_resized = transform(cloth_img).unsqueeze(0)
    cloth_mask = (cloth_img_resized.mean(dim=1, keepdim=True) > -0.9).float()

    # Get pose and segmentation
    pose_keypoints = get_pose_keypoints(human_img)
    parse = get_segmentation(human_img)
    img_agnostic = get_agnostic(human_img, parse, pose_keypoints)
    img_agnostic_tensor = transform(img_agnostic).unsqueeze(0)

    # Simulate pose image (placeholder)
    pose_rgb = human_img_resized

    # Part 1: Segmentation generation
    up = nn.Upsample(size=(opt.load_height, opt.load_width), mode='bilinear')
    gauss = tgm.image.GaussianBlur((15, 15), (3, 3))

    parse_agnostic = torch.zeros(1, opt.semantic_nc, opt.load_height, opt.load_width)  # Placeholder
    parse_agnostic_down = F.interpolate(parse_agnostic, size=(256, 192), mode='bilinear')
    pose_down = F.interpolate(pose_rgb, size=(256, 192), mode='bilinear')
    c_masked_down = F.interpolate(cloth_img_resized * cloth_mask, size=(256, 192), mode='bilinear')
    cm_down = F.interpolate(cloth_mask, size=(256, 192), mode='bilinear')
    seg_input = torch.cat((cm_down, c_masked_down, parse_agnostic_down, pose_down, gen_noise(cm_down.size())), dim=1)

    parse_pred_down = seg(seg_input)
    parse_pred = gauss(up(parse_pred_down))
    parse_pred = parse_pred.argmax(dim=1)[:, None]

    parse_old = torch.zeros(1, 13, opt.load_height, opt.load_width, dtype=torch.float)
    parse_old.scatter_(1, parse_pred, 1.0)

    labels = {
        0: ['background', [0]],
        1: ['paste', [2, 4, 7, 8, 9, 10, 11]],
        2: ['upper', [3]],
        3: ['hair', [1]],
        4: ['left_arm', [5]],
        5: ['right_arm', [6]],
        6: ['noise', [12]]
    }
    parse = torch.zeros(1, 7, opt.load_height, opt.load_width, dtype=torch.float)
    for j in range(len(labels)):
        for label in labels[j][1]:
            parse[:, j] += parse_old[:, label]

    # Part 2: Clothes deformation
    agnostic_gmm = F.interpolate(img_agnostic_tensor, size=(256, 192), mode='nearest')
    parse_cloth_gmm = F.interpolate(parse[:, 2:3], size=(256, 192), mode='nearest')
    pose_gmm = F.interpolate(pose_rgb, size=(256, 192), mode='nearest')
    c_gmm = F.interpolate(cloth_img_resized, size=(256, 192), mode='nearest')
    gmm_input = torch.cat((parse_cloth_gmm, pose_gmm, agnostic_gmm), dim=1)

    _, warped_grid = gmm(gmm_input, c_gmm)
    warped_c = F.grid_sample(cloth_img_resized, warped_grid, padding_mode='border')
    warped_cm = F.grid_sample(cloth_mask, warped_grid, padding_mode='border')

    # Part 3: Try-on synthesis
    misalign_mask = parse[:, 2:3] - warped_cm
    misalign_mask[misalign_mask < 0.0] = 0.0
    parse_div = torch.cat((parse, misalign_mask), dim=1)
    parse_div[:, 2:3] -= misalign_mask

    output = alias(torch.cat((img_agnostic_tensor, pose_rgb, warped_c), dim=1), parse, parse_div, misalign_mask)

    # Convert output to image
    output_img = output.squeeze(0).cpu().numpy()
    output_img = (output_img * 0.5 + 0.5).clip(0, 1)
    output_img = (output_img.transpose(1, 2, 0) * 255).astype(np.uint8)
    result_img = Image.fromarray(output_img)

    # Save to BytesIO for response
    img_byte_arr = BytesIO()
    result_img.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)

    return StreamingResponse(img_byte_arr, media_type="image/png")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)