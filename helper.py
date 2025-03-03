import os
from fastapi import HTTPException
from PIL import Image, ImageDraw
import numpy as np
from torchvision import transforms
from types import SimpleNamespace

def convert_to_grayscale(image_array):
    if len(image_array.shape) == 3 and image_array.shape[2] == 3:
        return np.mean(image_array, axis=2)  # Convert RGB to grayscale by averaging channels
    return image_array 

def get_opt():
    return SimpleNamespace(
        name="data",
        batch_size=1,
        workers=1,
        load_height=1024,
        load_width=768,
        shuffle=False,
        dataset_dir="./datasets/",
        dataset_mode="test",
        dataset_list="test_pairs.txt",
        checkpoint_dir="./checkpoints/",
        save_dir="./results/",
        display_freq=1,
        seg_checkpoint="seg_final.pth",
        gmm_checkpoint="gmm_final.pth",
        alias_checkpoint="alias_final.pth",
        semantic_nc=13,
        init_type="xavier",
        init_variance=0.02,
        grid_size=5,
        norm_G="spectralaliasinstance",
        ngf=64,
        num_upsampling_layers="most"
    )

#PARSE AGNOSTIC
def get_parse_agnostic( parse, pose_data):
    parse_array = np.array(parse)
    # parse_array = convert_to_grayscale(parse_array)
    parse_upper = ((parse_array == 5).astype(np.float32) +
                    (parse_array == 6).astype(np.float32) +
                    (parse_array == 7).astype(np.float32))
    parse_neck = (parse_array == 10).astype(np.float32)

    r = 10
    agnostic = parse.copy()

    # mask arms
    for parse_id, pose_ids in [(14, [2, 5, 6, 7]), (15, [5, 2, 3, 4])]:
        mask_arm = Image.new('L', (768, 1024), 'black')
        mask_arm_draw = ImageDraw.Draw(mask_arm)
        i_prev = pose_ids[0]
        for i in pose_ids[1:]:
            if (pose_data[i_prev, 0] == 0.0 and pose_data[i_prev, 1] == 0.0) or (pose_data[i, 0] == 0.0 and pose_data[i, 1] == 0.0):
                continue
            mask_arm_draw.line([tuple(pose_data[j]) for j in [i_prev, i]], 'white', width=r*10)
            pointx, pointy = pose_data[i]
            radius = r*4 if i == pose_ids[-1] else r*15
            mask_arm_draw.ellipse((pointx-radius, pointy-radius, pointx+radius, pointy+radius), 'white', 'white')
            i_prev = i   
        # parse_arm = (np.array(mask_arm) / 255) * (parse_array == parse_id).astype(np.float32)
        mask_arm_np = np.array(mask_arm)

        # Convert mask_arm to grayscale if it has 3 channels
        if len(mask_arm_np.shape) == 3 and mask_arm_np.shape[2] == 3:
            mask_arm_np = np.mean(mask_arm_np, axis=2)  # Convert RGB to grayscale

        mask_arm_np = mask_arm_np / 255  # Normalize

        # Ensure parse_array is single-channel
        parse_mask = (parse_array == parse_id).astype(np.float32)
        if len(parse_mask.shape) == 3:
            parse_mask = parse_mask[:, :, 0]  # Take only one channel

        # Now both arrays have shape (1024, 768)
        parse_arm = mask_arm_np * parse_mask
        agnostic.paste(0, None, Image.fromarray(np.uint8(parse_arm * 255), 'L'))

    # mask torso & neck
    agnostic.paste(0, None, Image.fromarray(np.uint8(parse_upper * 255), 'L'))
    agnostic.paste(0, None, Image.fromarray(np.uint8(parse_neck * 255), 'L'))

    return agnostic


#IMAGE AGNOSTIC
def get_img_agnostic( img, parse, pose_data):
    parse_array = np.array(parse)
    parse_head = ((parse_array == 4).astype(np.float32) +
                    (parse_array == 13).astype(np.float32))
    parse_lower = ((parse_array == 9).astype(np.float32) +
                    (parse_array == 12).astype(np.float32) +
                    (parse_array == 16).astype(np.float32) +
                    (parse_array == 17).astype(np.float32) +
                    (parse_array == 18).astype(np.float32) +
                    (parse_array == 19).astype(np.float32))

    r = 20
    agnostic = img.copy()
    agnostic_draw = ImageDraw.Draw(agnostic)

    length_a = np.linalg.norm(pose_data[5] - pose_data[2])
    length_b = np.linalg.norm(pose_data[12] - pose_data[9])
    point = (pose_data[9] + pose_data[12]) / 2
    pose_data[9] = point + (pose_data[9] - point) / length_b * length_a
    pose_data[12] = point + (pose_data[12] - point) / length_b * length_a

    # mask arms
    agnostic_draw.line([tuple(pose_data[i]) for i in [2, 5]], 'gray', width=r*10)
    for i in [2, 5]:
        pointx, pointy = pose_data[i]
        agnostic_draw.ellipse((pointx-r*5, pointy-r*5, pointx+r*5, pointy+r*5), 'gray', 'gray')
    for i in [3, 4, 6, 7]:
        if (pose_data[i - 1, 0] == 0.0 and pose_data[i - 1, 1] == 0.0) or (pose_data[i, 0] == 0.0 and pose_data[i, 1] == 0.0):
            continue
        agnostic_draw.line([tuple(pose_data[j]) for j in [i - 1, i]], 'gray', width=r*10)
        pointx, pointy = pose_data[i]
        agnostic_draw.ellipse((pointx-r*5, pointy-r*5, pointx+r*5, pointy+r*5), 'gray', 'gray')

    # mask torso
    for i in [9, 12]:
        pointx, pointy = pose_data[i]
        agnostic_draw.ellipse((pointx-r*3, pointy-r*6, pointx+r*3, pointy+r*6), 'gray', 'gray')
    agnostic_draw.line([tuple(pose_data[i]) for i in [2, 9]], 'gray', width=r*6)
    agnostic_draw.line([tuple(pose_data[i]) for i in [5, 12]], 'gray', width=r*6)
    agnostic_draw.line([tuple(pose_data[i]) for i in [9, 12]], 'gray', width=r*12)
    agnostic_draw.polygon([tuple(pose_data[i]) for i in [2, 5, 12, 9]], 'gray', 'gray')

    # mask neck
    pointx, pointy = pose_data[1]
    agnostic_draw.rectangle((pointx-r*7, pointy-r*7, pointx+r*7, pointy+r*7), 'gray', 'gray')
    agnostic.paste(img, None, Image.fromarray(np.uint8(parse_head * 255), 'L'))
    agnostic.paste(img, None, Image.fromarray(np.uint8(parse_lower * 255), 'L'))

    return agnostic


transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

def get_image_paths(human_filename: str) -> tuple[str, str, str, str]:
    # Define base directories
    base_image_dir = "datasets/test/image"
    image_path = os.path.join(base_image_dir, human_filename)
    
    # Check if the human image exists in the dataset
    if not os.path.exists(image_path):
        raise HTTPException(
            400,
            detail=f"Human image {human_filename} not found in dataset"
        )
    
    # Generate related file paths
    image = image_path
    image_parse = os.path.join(
        "datasets/test/image-parse",
        f"{human_filename.split('.')[0]}.png"
    )
    openpose_img = os.path.join(
        "datasets/test/openpose-img",
        f"{human_filename.split('.')[0]}_rendered.png"
    )
    openpose_json = os.path.join(
        "datasets/test/openpose-json",
        f"{human_filename.split('.')[0]}_keypoints.json"
    )
    
    # Verify all required files exist
    required_files = [image_parse, openpose_img, openpose_json]
    for file_path in required_files:
        if not os.path.exists(file_path):
            raise HTTPException(
                400,
                detail=f"Required preprocessing file not found: {file_path}"
            )
    
    return image, image_parse, openpose_img, openpose_json