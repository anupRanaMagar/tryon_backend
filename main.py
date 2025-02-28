from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
import os
import torch.nn.functional as F
from models import SegGenerator, GMM, ALIASGenerator
from utils import load_checkpoint
from helper import get_opt
from model_helper import segmentation_generation,clothes_deformation,try_on_synthesis
from load import load_data
from io import BytesIO
from PIL import Image

app = FastAPI()

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

@app.get("/")
async def read_root():
    return {"message": "Welcome to Virtual Tryon"}

@app.post("/tryon/")
# async def virtual_tryon(human_image: UploadFile = File(...), cloth_image: UploadFile = File(...)):
async def virtual_tryon():
    # cloth = Image.open(cloth_image.file)
    # #load data
    new_parse_agnostic_map, pose_rgb,cm,c, img_agnostic = load_data(openpose_img, openpose_json, image_parse,cloth, cloth_mask,image)

    # Part 1. Segmentation generation
    parse,pose_rgb = segmentation_generation(opt, new_parse_agnostic_map,pose_rgb,seg,cm,c)

    # Part 2. Clothes Deformation
    warped_c,warped_cm = clothes_deformation(img_agnostic, parse, pose_rgb,gmm,cm,c)

    # Part 3. Try-on synthesis
    im = try_on_synthesis(parse, pose_rgb, warped_c, img_agnostic, alias, warped_cm)

    img_bytes = BytesIO()
    im.save(img_bytes, format="JPEG")
    img_bytes.seek(0)

    return StreamingResponse(img_bytes, media_type="image/jpeg")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)