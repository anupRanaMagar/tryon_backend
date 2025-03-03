import os
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, UploadFile, File,HTTPException
from fastapi.responses import StreamingResponse
from models import SegGenerator, GMM, ALIASGenerator
from utils import load_checkpoint
from helper import get_opt,get_image_paths
from model_helper import segmentation_generation,clothes_deformation,try_on_synthesis
from load import load_data
from io import BytesIO
from PIL import Image
from cloth_mask_model import predic

app = FastAPI()

origins = [
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
async def virtual_tryon(
    human_image: UploadFile = File(...),
    cloth_image: UploadFile = File(...)
):
    try:
        if not human_image.content_type.startswith('image/') or \
           not cloth_image.content_type.startswith('image/'):
            raise HTTPException(400, detail="Invalid file type")

        human_filename = human_image.filename
        
        image_path = os.path.join("datasets/test/image", human_filename)
        if not os.path.exists(image_path):
            raise HTTPException(
                400,
                detail=f"Human image {human_filename} not found in dataset. Please ensure the image exists in datasets/test/image/"
            )
        image, image_parse, openpose_img, openpose_json = get_image_paths(human_filename)

        cloth = Image.open(cloth_image.file)
        cloth_mask = predic(cloth)

        new_parse_agnostic_map, pose_rgb, cm, c, img_agnostic = load_data(
            openpose_img, openpose_json, image_parse, cloth, cloth_mask, image
        )
        
        parse, pose_rgb = segmentation_generation(opt, new_parse_agnostic_map, pose_rgb, seg, cm, c)
        warped_c, warped_cm = clothes_deformation(img_agnostic, parse, pose_rgb, gmm, cm, c)
        im = try_on_synthesis(parse, pose_rgb, warped_c, img_agnostic, alias, warped_cm)

        img_bytes = BytesIO()
        im.save(img_bytes, format="JPEG")
        img_bytes.seek(0)
        
        return StreamingResponse(img_bytes, media_type="image/jpeg")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)