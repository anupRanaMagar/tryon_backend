from PIL import Image,ImageDraw
from torchvision import transforms
import json
import os
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchgeometry as tgm
from models import SegGenerator, GMM, ALIASGenerator
import argparse
from utils import gen_noise, load_checkpoint, save_images
from helper import get_opt,get_parse_agnostic,get_img_agnostic,transform


image = "datasets/test/image/00891_00.jpg"
cloth = "datasets/test/cloth/07429_00.jpg"
cloth_mask = "datasets/test/cloth-mask/07429_00.jpg"
image_parse = "datasets/test/image-parse/00891_00.png"
openpose_img = "datasets/test/openpose-img/00891_00_rendered.png"
openpose_json = "datasets/test/openpose-json/00891_00_keypoints.json"

opt = get_opt()

seg = SegGenerator(opt, input_nc=opt.semantic_nc + 8, output_nc=opt.semantic_nc)
gmm = GMM(opt, inputA_nc=7, inputB_nc=3)
opt.semantic_nc = 7
alias = ALIASGenerator(opt, input_nc=9)
opt.semantic_nc = 13

load_checkpoint(seg, os.path.join(opt.checkpoint_dir, opt.seg_checkpoint))
load_checkpoint(gmm, os.path.join(opt.checkpoint_dir, opt.gmm_checkpoint))
load_checkpoint(alias, os.path.join(opt.checkpoint_dir, opt.alias_checkpoint))

seg.eval()
gmm.eval()
alias.eval()

#load pose image
pose_rgb = Image.open(openpose_img)
pose_rg =transforms.Resize(768, interpolation=2)(pose_rgb)
pose_rgb = transform(pose_rgb)

with open(openpose_json, 'r') as f:
    pose_label = json.load(f)
    pose_data = pose_label['people'][0]['pose_keypoints_2d']
    pose_data = np.array(pose_data)
    pose_data = pose_data.reshape((-1, 3))[:, :2]

#load parsing image
parse = Image.open(image_parse)
parse = transforms.Resize(768, interpolation=0)(parse)
parse_agnostic = get_parse_agnostic(parse, pose_data)
parse_agnostic = torch.from_numpy(np.array(parse_agnostic)[None]).long()

labels = {
            0: ['background', [0, 10]],
            1: ['hair', [1, 2]],
            2: ['face', [4, 13]],
            3: ['upper', [5, 6, 7]],
            4: ['bottom', [9, 12]],
            5: ['left_arm', [14]],
            6: ['right_arm', [15]],
            7: ['left_leg', [16]],
            8: ['right_leg', [17]],
            9: ['left_shoe', [18]],
            10: ['right_shoe', [19]],
            11: ['socks', [8]],
            12: ['noise', [3, 11]]
        }

parse_agnostic_map = torch.zeros(20,1024,768, dtype=torch.float)
parse_agnostic_map.scatter_(0,parse_agnostic,1.0)
new_parse_agnostic_map = torch.zeros(13,1024,768, dtype=torch.float)
for i in range(len(labels)):
            for label in labels[i][1]:
                new_parse_agnostic_map[i] += parse_agnostic_map[label]


#load person image
img = Image.open(image)
img = transforms.Resize(768, interpolation=2)(img)
img_agnostic = get_img_agnostic(img, parse, pose_data)
img = transform(img)
img_agnostic = transform(img_agnostic)


c = Image.open(cloth).convert('RGB')
c = transforms.Resize(768, interpolation=2)(c)
cm = Image.open(cloth_mask)
cm = transforms.Resize(768, interpolation=0)(cm)


c = transform(c)
cm_array = np.array(cm)
cm_array = (cm_array >=128).astype(np.float32)
cm = torch.from_numpy(cm_array)
cm.unsqueeze_(0)

up = nn.Upsample(size=(opt.load_height, opt.load_width), mode='bilinear')
gauss = tgm.image.GaussianBlur((15, 15), (3, 3))

# Part 1. Segmentation generation
new_parse_agnostic_map = new_parse_agnostic_map.unsqueeze(0) 
parse_agnostic_down = F.interpolate(new_parse_agnostic_map, size=(256, 192), mode='bilinear')
pose_rgb = pose_rgb.unsqueeze(0)
pose_down = F.interpolate(pose_rgb, size=(256, 192), mode='bilinear')
c_masked_down = F.interpolate((c * cm).unsqueeze(0), size=(256, 192), mode='bilinear')
cm_down = F.interpolate(cm.unsqueeze(0), size=(256, 192), mode='bilinear')


seg_input = torch.cat((cm_down, c_masked_down, parse_agnostic_down, pose_down[:1], gen_noise(cm_down.size())), dim=1)

parse_pred_down = seg(seg_input)
parse_pred = gauss(up(parse_pred_down))
parse_pred = parse_pred.argmax(dim=1)[:, None]

parse_old = torch.zeros(parse_pred.size(0), 13, opt.load_height, opt.load_width, dtype=torch.float)
parse_old.scatter_(1, parse_pred, 1.0)

labels = {
    0:  ['background',  [0]],
    1:  ['paste',       [2, 4, 7, 8, 9, 10, 11]],
    2:  ['upper',       [3]],
    3:  ['hair',        [1]],
    4:  ['left_arm',    [5]],
    5:  ['right_arm',   [6]],
    6:  ['noise',       [12]]
}
parse = torch.zeros(parse_pred.size(0), 7, opt.load_height, opt.load_width, dtype=torch.float)
for j in range(len(labels)):
    for label in labels[j][1]:
        parse[:, j] += parse_old[:, label]

# Part 2. Clothes Deformation

agnostic_gmm = F.interpolate(img_agnostic.unsqueeze(0), size=(256, 192), mode='nearest')
parse_cloth_gmm = F.interpolate(parse[:, 2:3], size=(256, 192), mode='nearest')
pose_gmm = F.interpolate(pose_rgb, size=(256, 192), mode='nearest')
c_gmm = F.interpolate(c.unsqueeze(0), size=(256, 192), mode='nearest')
gmm_input = torch.cat((parse_cloth_gmm, pose_gmm, agnostic_gmm), dim=1)

_, warped_grid = gmm(gmm_input, c_gmm)
c= c.unsqueeze(0)
print("WARPED GRID",warped_grid.size())
print("C shape:", c.size())
cm = cm.unsqueeze(0)
warped_c = F.grid_sample(c, warped_grid, padding_mode='border')
warped_cm = F.grid_sample(cm, warped_grid, padding_mode='border')

# Part 3. Try-on synthesis
misalign_mask = parse[:, 2:3] - warped_cm
misalign_mask[misalign_mask < 0.0] = 0.0
parse_div = torch.cat((parse, misalign_mask), dim=1)
parse_div[:, 2:3] -= misalign_mask

output = alias(torch.cat((img_agnostic.unsqueeze(0), pose_rgb, warped_c), dim=1), parse, parse_div, misalign_mask)


tensor = (output.clone()+1)*0.5 * 255
tensor = tensor.cpu().clamp(0,255)
try:
    array = tensor.numpy().astype('uint8')
except:
    array = tensor.detach().numpy().astype('uint8')

if array.shape[0] == 1:
    array = array.squeeze(0)
elif array.shape[0] == 3:
    array = array.swapaxes(0, 1).swapaxes(1, 2)

print("ARRAY SHAPE", array.shape)
array = array.transpose(1, 2, 0)
im = Image.fromarray(array)
im.save("finaloutput.jpg", format='JPEG')
