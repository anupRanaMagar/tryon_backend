import torch
import torch.nn.functional as F
import torchgeometry as tgm
from torch import nn
from utils import gen_noise




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
    
    return parse, 


