import torch
import torch.nn.functional as F
import torchgeometry as tgm
from torch import nn
from utils import gen_noise
from PIL import Image

def segmentation_generation(opt, new_parse_agnostic_map,pose_rgb,seg,cm,c):
    up = nn.Upsample(size=(opt.load_height, opt.load_width), mode='bilinear')
    gauss = tgm.image.GaussianBlur((15, 15), (3, 3))
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
    
    return parse,pose_rgb


def clothes_deformation(img_agnostic, parse, pose_rgb,gmm,cm,c):
    agnostic_gmm = F.interpolate(img_agnostic.unsqueeze(0), size=(256, 192), mode='nearest')
    parse_cloth_gmm = F.interpolate(parse[:, 2:3], size=(256, 192), mode='nearest')
    pose_gmm = F.interpolate(pose_rgb, size=(256, 192), mode='nearest')
    c_gmm = F.interpolate(c.unsqueeze(0), size=(256, 192), mode='nearest')
    gmm_input = torch.cat((parse_cloth_gmm, pose_gmm, agnostic_gmm), dim=1)

    _, warped_grid = gmm(gmm_input, c_gmm)
    c= c.unsqueeze(0)
    cm = cm.unsqueeze(0)
    warped_c = F.grid_sample(c, warped_grid, padding_mode='border')
    warped_cm = F.grid_sample(cm, warped_grid, padding_mode='border')
    return warped_c,warped_cm

def try_on_synthesis(parse, pose_rgb, warped_c, img_agnostic, alias, warped_cm):
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

    array = array.transpose(1, 2, 0)
    im = Image.fromarray(array)
    return im
    
    





