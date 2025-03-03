from PIL import Image
from torchvision import transforms
import json
import torch
import numpy as np
from helper import get_parse_agnostic,get_img_agnostic,transform

def convert_to_grayscale(image_array):
    if len(image_array.shape) == 3 and image_array.shape[2] == 3:
        return np.mean(image_array, axis=2)  # Convert RGB to grayscale by averaging channels
    return image_array 

def load_data(openpose_img, openpose_json, image_parse,cloth, cloth_mask,image):
    pose_rgb = Image.open(openpose_img)
    pose_rgb = transforms.Resize((1024, 768), interpolation=Image.BILINEAR)(pose_rgb)
    # pose_rgb =transforms.Resize(768, interpolation=2)(pose_rgb)
    pose_rgb = transform(pose_rgb)

    with open(openpose_json, 'r') as f:
        pose_label = json.load(f)
        pose_data = pose_label['people'][0]['pose_keypoints_2d']
        pose_data = np.array(pose_data)
        pose_data = pose_data.reshape((-1, 3))[:, :2]

    #load parsing image
    parse = Image.open(image_parse)
    parse = parse.convert("P")
    parse = transforms.Resize((1024, 768), interpolation=Image.NEAREST)(parse)
    # parse = transforms.Resize(768, interpolation=0)(parse)
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
    for i, (_, indices) in labels.items():
            for label in indices:
                new_parse_agnostic_map[i] += parse_agnostic_map[label]


    #load person image
    img = Image.open(image)
    img = transforms.Resize(768, interpolation=2)(img)
    img_agnostic = get_img_agnostic(img, parse, pose_data)
    img = transform(img)
    img_agnostic = transform(img_agnostic)


    # c = Image.open(cloth).convert('RGB')
    c = cloth.convert('RGB')
    c = c.resize((768,1024), Image.BICUBIC)
    c = transforms.Resize(768, interpolation=2)(c)
    # cm = Image.open(cloth_mask)
    cm = cloth_mask
    cm = transforms.Resize(768, interpolation=0)(cm)


    c = transform(c)
    cm_array = np.array(cm)
    cm_array = (cm_array >=128).astype(np.float32)
    cm = torch.from_numpy(cm_array)
    cm.unsqueeze_(0)

    return new_parse_agnostic_map, pose_rgb,cm,c, img_agnostic