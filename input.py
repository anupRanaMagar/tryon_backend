import os
import torch.nn.functional as F
from models import SegGenerator, GMM, ALIASGenerator
from utils import load_checkpoint
from helper import get_opt
from model_helper import segmentation_generation,clothes_deformation,try_on_synthesis
from load import load_data
from cloth_mask_model import predic 
from PIL import Image


# image = "datasets/test/image/08909_00.jpg"
cloth = "datasets/test/cloth/07429_00.jpg"
cloth_mask = "datasets/test/cloth-mask/07429_00.jpg"
# cloth_mask = 'output_mask.jpg'
# image_parse = "girl1.png"
# image_parse = "datasets/test/image-parse/08909_00.png"
# openpose_img = "datasets/test/openpose-img/08909_00_rendered.png"
# openpose_json = "datasets/test/openpose-json/10549_00_keypoints.json"

image = "a3.jpg"
# cloth = "datasets/test/cloth/07429_00.jpg"
# cloth_mask = "datasets/test/cloth-mask/07429_00.jpg"
# image_parse = "girl1.png"
image_parse = "palette.png"
# image_parse = "ishan6.png"
# image_parse = "datasets/test/image-parse/08909_00.png"
openpose_img = "a2.png"
openpose_json = "datasets/test/openpose-json/10549_00_keypoints.json"

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


cloth = Image.open(cloth)
cloth_mask = predic(cloth)
# #load data
new_parse_agnostic_map, pose_rgb,cm,c, img_agnostic = load_data(openpose_img, openpose_json, image_parse,cloth, cloth_mask,image)

# Part 1. Segmentation generation
parse,pose_rgb = segmentation_generation(opt, new_parse_agnostic_map,pose_rgb,seg,cm,c)

# Part 2. Clothes Deformation
warped_c,warped_cm = clothes_deformation(img_agnostic, parse, pose_rgb,gmm,cm,c)

# Part 3.  Try-on synthesis
im = try_on_synthesis(parse, pose_rgb, warped_c, img_agnostic, alias, warped_cm)
im.save("finaloutput.jpg", format='JPEG')
