import configparser
import os
import random

import torch
from torch.utils import data
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from einops import rearrange, repeat

from glob import glob

def normalize_input(
    item, 
    mean=[0.5, 0.5, 0.5], # Imagenet [0.485, 0.456, 0.406]
    std=[0.5, 0.5, 0.5], # Imagenet [0.229, 0.224, 0.225]
    use_simple_norm=False
):
    if item.dtype == torch.uint8 and not use_simple_norm:
        item = rearrange(item, 'f c h w -> f h w c')
        
        item = item.float() / 255.0
        mean = torch.tensor(mean)
        std = torch.tensor(std)

        out = rearrange((item - mean) / std, 'f h w c -> f c h w')
        
        return out
    else:
        
        item = rearrange(item, 'f c h w -> f h w c')
        return  rearrange(item / 127.5 - 1.0, 'f h w c -> f c h w')

def generate_prompt(K_indices, groundtruth_txt, meta_info_ini):
    # Parse groundtruth.txt
    with open(groundtruth_txt, 'r') as f:
        bbox_lines = f.readlines()
    selected_bboxes = [bbox_lines[i].strip() for i in K_indices]

    ori_bbox = []
    
    config = configparser.ConfigParser()
    config.read(meta_info_ini)
    img_w, img_h = tuple(map(int, config["METAINFO"]["resolution"][1:-1].split(', ')))

    for i, bbox in enumerate(selected_bboxes):
        x, y, w, h = map(float, bbox.split(','))
        x1, y1, x2, y2 = x / img_w, y / img_h, (x + w) / img_w, (y + h) / img_h
        x1, y1, x2, y2 = map(lambda v: min(max(v, 0.0), 1.0), [x1, y1, x2, y2])
        ori_bbox.append([x1, y1, x2, y2, 0])

    def get_grid_index(x, y, total_columns=192):
        grid_size = 2  # Since the image is divided into 192x192 grids on a 384x384 image
        grid_x = int(x / grid_size)
        grid_y = int(y / grid_size)
        return grid_y * total_columns + grid_x

    # Parse meta_info.ini
    meta_info = {}
    with open(meta_info_ini, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if ':' in line and 'class' in line:
                key, value = line.split(':')
                key = key.strip()
                value = value.strip()
                meta_info[key] = value
    
    # Generate the prompt
    object_class = meta_info["object_class"]
    motion_class = meta_info["motion_class"]

    prompt = f"A {object_class} that is {motion_class}. "
    seg_phrase = f"{object_class}"
    # for i, index in enumerate(K_indices):
    #     loc_index_tl, loc_index_br = get_grid_index(resize_bbox[i][0], resize_bbox[i][1]), get_grid_index(resize_bbox[i][2], resize_bbox[i][3])
    #     prompt += f"(<f{i}>, <l{loc_index_tl}>, <l{loc_index_br}>) "
    return prompt, ori_bbox, seg_phrase

class GOT10KDataset(Dataset):
    def __init__(
        self,
        width: int = 256,
        height: int = 256,
        n_sample_frames: int = 8,
        path: str = "./data",
        fallback_prompt: str = "",
        **kwargs
    ):
        self.fallback_prompt = fallback_prompt

        # List all folders (each folder corresponds to a video)
        frame_folders = glob(f"{path}/*")
        frame_folders.sort()
        frame_folders = frame_folders[:-1]
        # drop list.txt
        drop_folder = ['001898', '009302', '008628', '001858', '001849', '009058', '009065',
                        '001665', '004419', '009059', '009211', '009210', '009032', '009102',
                          '002446', '008925', '001867', '009274', '009323', '002449', '009031',
                            '009094', '005912', '007643', '007788', '008917', '009214', '007610',
                              '009320', '007645', '009027', '008721', '008931', '008630']
        self.frame_folders = [ok_name for ok_name in frame_folders if not any(drop in ok_name for drop in drop_folder)]
        self.width = width
        self.height = height
        self.n_sample_frames = n_sample_frames

    def get_frame_buckets(self, frame):
        w, h = frame.size
        # width, height = sensible_buckets(self.width, self.height, w, h)
        width, height = self.width, self.height

        w_scale = width / w
        h_scale = height / h
        resize = T.transforms.Resize((height, width), antialias=True)

        return resize, w_scale, h_scale

    def get_frame_batch(self, folder_path, interval=None):
        # Get all jpg images in the folder
        if interval is None:
            # interval = random.randint(1, 2)
            interval = 1
        frame_files = sorted(glob(f"{folder_path}/*.jpg"))
        
        # Sample frames at the given interval
        sampled_frame_files = frame_files[::interval]
        indices = list(range(0, len(frame_files), interval))
        
        # Ensure we don't sample more than the required number of frames
        if len(sampled_frame_files) >= self.n_sample_frames:
            # random start
            start_idx = random.randint(0, len(sampled_frame_files)-self.n_sample_frames)
            sampled_frame_files = sampled_frame_files[start_idx : start_idx+self.n_sample_frames]

        # Read the first frame to determine the original size and get the resize transform
        f_sample = Image.open(sampled_frame_files[0])
        resize_transform, w_scale, h_scale = self.get_frame_buckets(f_sample)

        # Convert list of frames to a tensor
        return sampled_frame_files, resize_transform, w_scale, h_scale, indices[start_idx : start_idx+self.n_sample_frames]

    def get_prompt(self, folder_path, k_index):
        # Check if groundtruth.txt exists
        groundtruth_txt_path = os.path.join(folder_path, "groundtruth.txt")
        meta_info_ini_path = os.path.join(folder_path, "meta_info.ini")
        
        if os.path.exists(groundtruth_txt_path) and os.path.exists(meta_info_ini_path):
            # FOOL! You must use the true index!
            return generate_prompt(k_index, groundtruth_txt_path, meta_info_ini_path)
        else:
            return self.fallback_prompt

    @staticmethod
    def __getname__(): return 'frame'

    def __len__(self):
        return len(self.frame_folders)

    def __getitem__(self, index):
        sampled_frame_files, resize_transform, w_scale, h_scale, k_index  = self.get_frame_batch(self.frame_folders[index])
        prompt, ori_bboxes, seg_phrase = self.get_prompt(self.frame_folders[index], k_index)
        
        # Define the transform using albumentations
        crop_transform = A.Compose([
            A.Resize(height=self.height, width=self.width),
            ToTensorV2()  # Convert to PyTorch tensor
        ], bbox_params=A.BboxParams(format="albumentations"),)
        
        # Apply crop transform to each frame and update bboxes
        frames = []
        resize_bboxes = []
        for idx, frame_file in enumerate(sampled_frame_files):
            image = np.array(Image.open(frame_file).convert("RGB")).astype(np.uint8)
            transformed = crop_transform(image=image, bboxes=[ori_bboxes[idx],])
            frames.append(transformed['image'])
            resize_bboxes.append(transformed['bboxes'][0])  # Update bboxes to match cropped image

        frames = torch.stack(frames)

        def dummy_normalize_bboxes(bboxes):
            return [[xmin, ymin, xmax, ymax] for xmin, ymin, xmax, ymax, dummy_class in bboxes]
        
        def draw_bboxes_on_frames(frames, bboxes):
            bboxes = np.array(bboxes)
            from PIL import Image, ImageDraw
            _, _, height, width = frames.shape

            frames_np = (frames).numpy().transpose(0, 2, 3, 1)
            frames_np = frames_np.astype(np.uint8)
            
            cnt = 0
            for frame, bbox in zip(frames_np, bboxes):
                img = Image.fromarray(frame)
                draw = ImageDraw.Draw(img)
                xmin, ymin, xmax, ymax = bbox[0], bbox[1], bbox[2], bbox[3]
                xmin *= 480
                xmax *= 480
                ymin *= 320
                ymax *= 320
                draw.rectangle([(xmin, ymin), (xmax, ymax)], outline=(0, 255, 0), width=2)
                img.save(f'test{cnt}.jpg')
                cnt += 1


        bboxes = torch.Tensor(dummy_normalize_bboxes(resize_bboxes))
        bboxes = torch.clamp(bboxes, min=0.0, max=1.0)
        draw_bboxes_on_frames(frames, bboxes)

        return {
            "pixel_values": normalize_input(frames),
            # "prompt_ids": prompt_ids[0],
            "text_prompt": prompt,
            "seg_phrase": seg_phrase,
            'bboxes': bboxes, # TODO: Add bounding boxes to the dataset
            'dataset': self.__getname__()
        }

if __name__ == "__main__":
    dataset = GOT10KDataset(path='/GOT10K/train', width=480, height=320, n_sample_frames=8, fallback_prompt="A person that is walking.")
    idx = 0
    train_dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=1,
        shuffle=True,
        num_workers=0,
    )
    for step, batch in enumerate(train_dataloader):
        print(batch['text_prompt'])
        break
        # continue