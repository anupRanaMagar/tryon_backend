import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import json
import os.path as osp
from PIL import Image, ImageDraw
from test import get_opt

image_path = 'datasets/test/image/00891_00.jpg'
cloth_path = 'datasets/test/cloth/01430_00.jpg'


class VITONDataset(data.Dataset):
    def __init__(self, opt):
        super(VITONDataset, self).__init__()
        self.load_height = opt.load_height
        self.load_width = opt.load_width
        self.semantic_nc = opt.semantic_nc
        self.data_path = osp.join(opt.dataset_dir, opt.dataset_mode)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # Load single image name and cloth name from options
        self.img_name = "00891_00.jpg"  # Single image
        self.c_name = "01430_00.jpg"      # Single cloth

    def get_parse_agnostic(self, parse, pose_data):
        parse_array = np.array(parse)
        parse_upper = ((parse_array == 5).astype(np.float32) +
                       (parse_array == 6).astype(np.float32) +
                       (parse_array == 7).astype(np.float32))
        parse_neck = (parse_array == 10).astype(np.float32)

        r = 10
        agnostic = parse.copy()

        # mask arms
        for parse_id, pose_ids in [(14, [2, 5, 6, 7]), (15, [5, 2, 3, 4])]:
            mask_arm = Image.new('L', (self.load_width, self.load_height), 'black')
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
            parse_arm = (np.array(mask_arm) / 255) * (parse_array == parse_id).astype(np.float32)
            agnostic.paste(0, None, Image.fromarray(np.uint8(parse_arm * 255), 'L'))

        # mask torso & neck
        agnostic.paste(0, None, Image.fromarray(np.uint8(parse_upper * 255), 'L'))
        agnostic.paste(0, None, Image.fromarray(np.uint8(parse_neck * 255), 'L'))

        return agnostic

    def __getitem__(self, index):
        # Load person image
        # img_path = osp.join(self.data_path, 'image', self.img_name)
        img_path = image_path
        img = Image.open(img_path).convert('RGB')
        img = transforms.Resize(self.load_width, interpolation=2)(img)

        # Load cloth and cloth mask
        # c_path = osp.join(self.data_path, 'cloth', self.c_name)
        c_path = cloth_path
        c = Image.open(c_path).convert('RGB')
        c = transforms.Resize(self.load_width, interpolation=2)(c)
        cm_path = osp.join(self.data_path, 'cloth-mask', self.c_name)
        cm = Image.open(cm_path).convert('L')
        cm = transforms.Resize(self.load_width, interpolation=0)(cm)

        # Apply transforms
        img = self.transform(img)
        c = self.transform(c)
        cm_array = np.array(cm)
        cm_array = (cm_array >= 128).astype(np.float32)
        cm = torch.from_numpy(cm_array).unsqueeze(0)  # Add channel dimension

        # Load pose image
        pose_path = osp.join(self.data_path, 'openpose-img', self.img_name.replace('.jpg', '_rendered.png'))
        pose_rgb = Image.open(pose_path).convert('RGB')
        pose_rgb = transforms.Resize(self.load_width, interpolation=2)(pose_rgb)
        pose_rgb = self.transform(pose_rgb)

        pose_json_path = osp.join(self.data_path, 'openpose-json', self.img_name.replace('.jpg', '_keypoints.json'))
        with open(pose_json_path, 'r') as f:
            pose_label = json.load(f)
            pose_data = np.array(pose_label['people'][0]['pose_keypoints_2d']).reshape((-1, 3))[:, :2]

        # Load parsing image
        parse_path = osp.join(self.data_path, 'image-parse', self.img_name.replace('.jpg', '.png'))
        parse = Image.open(parse_path).convert('L')
        parse = transforms.Resize(self.load_width, interpolation=0)(parse)
        parse_agnostic = self.get_parse_agnostic(parse, pose_data)
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
        parse_agnostic_map = torch.zeros(20, self.load_height, self.load_width, dtype=torch.float)
        parse_agnostic_map.scatter_(0, parse_agnostic, 1.0)
        new_parse_agnostic_map = torch.zeros(self.semantic_nc, self.load_height, self.load_width, dtype=torch.float)
        for i in range(len(labels)):
            for label in labels[i][1]:
                new_parse_agnostic_map[i] += parse_agnostic_map[label]

        result = {
            'img_name': self.img_name,
            'c_name': self.c_name,
            'img': img,
            'parse_agnostic': new_parse_agnostic_map,
            'pose': pose_rgb,
            'cloth': c,
            'cloth_mask': cm,
        }
        return result

    def __len__(self):
        return 1  # Only one image
    



class VITONDataLoader:
    def __init__(self, opt, dataset):
        super(VITONDataLoader, self).__init__()

        if opt.shuffle:
            train_sampler = data.sampler.RandomSampler(dataset)
        else:
            train_sampler = None

        self.data_loader = data.DataLoader(
                dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
                num_workers=opt.workers, pin_memory=True, drop_last=True, sampler=train_sampler
        )
        self.dataset = dataset
        self.data_iter = self.data_loader.__iter__()

    def next_batch(self):
        try:
            batch = self.data_iter.__next__()
        except StopIteration:
            self.data_iter = self.data_loader.__iter__()
            batch = self.data_iter.__next__()

        return batch

opt = get_opt()
test_dataset = VITONDataset(opt)

Namespace(name='anup', batch_size=1, workers=1, load_height=1024, load_width=768, shuffle=False, dataset_dir='./datasets/', dataset_mode='test', dataset_list='test_pairs.txt', checkpoint_dir='./checkpoints/', save_dir='./results/', display_freq=1, seg_checkpoint='seg_final.pth', gmm_checkpoint='gmm_final.pth', alias_checkpoint='alias_final.pth', semantic_nc=13, init_type='xavier', init_variance=0.02, grid_size=5, norm_G='spectralaliasinstance', ngf=64, num_upsampling_layers='most')