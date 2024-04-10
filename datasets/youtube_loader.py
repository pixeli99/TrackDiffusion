# Copyright (c) OpenMMLab. All rights reserved.
import os
import mmengine

from mmtrack.registry import DATASETS
from mmtrack.datasets import BaseVideoDataset

import torch

from typing import Any, List, Tuple
import random
import json

import torch
from torchvision.transforms.functional import to_tensor
from PIL import Image, ImageDraw

color_list = [
    (220, 20, 60),    # Bright red
    (0, 0, 255),      # Dark blue
    (106, 0, 228),    # Vivid purple
    (246, 0, 122),    # Bright pink
    (250, 170, 30),   # Bright orange
    (220, 220, 0),    # Yellow
    (165, 42, 42),    # Brown
    (0, 226, 252),    # Bright cyan
    (182, 182, 255),  # Light periwinkle
    (120, 166, 157),  # Soft teal
    (255, 179, 240),  # Pale pink
    (0, 125, 92),     # Sea green
    (0, 220, 176),    # Turquoise
    (255, 99, 164),   # Soft magenta
    (45, 89, 255),    # Bright periwinkle
    (134, 134, 103),  # Olive green
    (197, 226, 255),  # Light sky blue
    (207, 138, 255),  # Lavender
    (74, 65, 105),    # Dark lavender
    (255, 109, 65),   # Bright coral
    (0, 143, 149),    # Dark turquoise
    (209, 99, 106),   # Soft red
    (227, 255, 205),  # Very pale lime
    (163, 255, 0),    # Neon green
    (183, 130, 88),   # Tan
    (166, 74, 118),   # Mauve
    (65, 70, 15),     # Dark olive
    (191, 162, 208),  # Soft purple
    (142, 108, 45)    # Umber
]

def draw_bboxes_on_image(images, bboxes, image_size, color_set):
    """
    :param images: Tensor of images, shape [batch_size, channels, height, width].
    :param bboxes: Tensor of normalized bbox coordinates, shape [batch_size, num_bboxes, 4].
    :param image_size: Tuple of (height, width) representing the original image size.
    """
    drawn_images = []
    for img, bbox in zip(images, bboxes):
        img = img.permute(1, 2, 0)
        img = img.byte().numpy()
        img = Image.fromarray(img)
        draw = ImageDraw.Draw(img)
        idx = 0
        for box in bbox:
            x_min, y_min, x_max, y_max = box
            x_min = x_min * image_size[1]
            y_min = y_min * image_size[0]
            x_max = x_max * image_size[1]
            y_max = y_max * image_size[0]
            
            draw.rectangle([x_min, y_min, x_max, y_max], outline=tuple(color_set[idx]), width=4)
            idx += 1

        drawn_images.append(img)
    
    return drawn_images

def vis_collate_fn(batch):
    # Convert BGR to RGB
    video_pixel_values = []
    video_bboxes = []
    video_cls = []
    video_masks = []
    
    ins_ids = []
    for data in batch:
        ins_id = data['data_samples'].gt_instances.instances_id
        ins_ids.append(ins_id[None])
    ins_ids = torch.cat(ins_ids, dim=-1)
    unique_ins_ids = torch.unique(ins_ids)
    id_to_index_map = {int(id): index for index, id in enumerate(unique_ins_ids)}

    for data in batch:
        ori_image = data['inputs']['img'][:, [2, 1, 0], :, :]
        # Normalize the image
        ori_image = ori_image / 127.5 - 1
        video_pixel_values.append(ori_image.squeeze(0))

        _, _, height, width = ori_image.shape
        bboxes = data['data_samples'].gt_instances.bboxes
        ins_id = data['data_samples'].gt_instances.instances_id
        
        bboxes[:, [0, 2]] /= width   # x_min and x_max
        bboxes[:, [1, 3]] /= height  # y_min and y_max
        
        ins_id = data['data_samples'].gt_instances.instances_id
        # padding none
        max_length = 20
        num_objects = bboxes.shape[0]

        bboxes_padded = torch.zeros(max_length, 4)
        classes_name_idx = torch.full((max_length,), 1000)
        mask = torch.zeros(max_length,)

        for idx, _ins_id in enumerate(ins_id):
            insertion_idx = id_to_index_map.get(int(_ins_id))
            if insertion_idx is None or insertion_idx >= max_length:
                continue
            bboxes_padded[insertion_idx] = bboxes[idx]
            classes_name_idx[insertion_idx] = data['data_samples'].gt_instances.labels[idx]
            mask[insertion_idx] = 1

        video_bboxes.append(bboxes_padded)
        video_cls.append(classes_name_idx.int())
        video_masks.append(mask)
        video_name = data['data_samples'].img_path.split('/')[-2]

    video_pixel_values = torch.stack(video_pixel_values)
    video_bboxes = torch.stack(video_bboxes)
    video_cls = torch.stack(video_cls)
    video_masks = torch.stack(video_masks)

    return video_pixel_values, video_bboxes, video_cls, video_masks, video_name

def got_collate_fn(batch):
    # Convert BGR to RGB
    video_pixel_values = []
    video_bboxes = []
    video_cls = []
    video_masks = []
    
    ins_ids = []
    for data in batch:
        ins_id = data['data_samples'].gt_instances.instances_id
        ins_ids.append(ins_id[None])
    ins_ids = torch.cat(ins_ids, dim=-1)
    unique_ins_ids = torch.unique(ins_ids)
    id_to_index_map = {int(id): index for index, id in enumerate(unique_ins_ids)}

    for data in batch:
        ori_image = data['inputs']['img'][:, [2, 1, 0], :, :]
        # Normalize the image
        ori_image = ori_image / 127.5 - 1
        video_pixel_values.append(ori_image.squeeze(0))

        _, _, height, width = ori_image.shape
        bboxes = data['data_samples'].gt_instances.bboxes
        ins_id = data['data_samples'].gt_instances.instances_id
        
        bboxes[:, [0, 2]] /= width   # x_min and x_max
        bboxes[:, [1, 3]] /= height  # y_min and y_max
        
        ins_id = data['data_samples'].gt_instances.instances_id
        # padding none
        max_length = 1
        num_objects = bboxes.shape[0]

        bboxes_padded = torch.zeros(max_length, 4)
        classes_name_idx = torch.full((max_length,), 1000)
        mask = torch.zeros(max_length,)

        for idx, _ins_id in enumerate(ins_id):
            insertion_idx = id_to_index_map.get(int(_ins_id))
            bboxes_padded[insertion_idx] = bboxes[idx]
            classes_name_idx[insertion_idx] = data['data_samples'].gt_instances.labels[idx]
            mask[insertion_idx] = 1

        video_bboxes.append(bboxes_padded)
        video_cls.append(classes_name_idx.int())
        video_masks.append(mask)
        video_name = data['data_samples'].img_path.split('/')[3]

    video_pixel_values = torch.stack(video_pixel_values)
    video_bboxes = torch.stack(video_bboxes)
    video_cls = torch.stack(video_cls)
    video_masks = torch.stack(video_masks)

    return video_pixel_values, video_bboxes, video_cls, video_masks, video_name      
                
@DATASETS.register_module()
class VISDataset(BaseVideoDataset):
    def __init__(self, dataset_version, interval=1, *args, **kwargs):
        self.set_dataset_classes(dataset_version)
        super().__init__(*args, **kwargs)
        
        if dataset_version in ['2021', '2019']:
            with open('/youtubevis_caption.json', 'r', encoding='utf-8') as file:
                caption_data = json.load(file)
        elif dataset_version == 'got10k':
            with open('/got_train.json', 'r', encoding='utf-8') as file:
                caption_data = json.load(file)
        else:
            caption_data = None
        self.caption_data = caption_data
        self.interval = interval
        self.dataset_version = dataset_version


    @classmethod
    def set_dataset_classes(cls, dataset_version) -> None:
        """Pass the category of the corresponding year to metainfo.

        Args:
            dataset_version (str): Select dataset year version.
        """
        if dataset_version == '2021':
            CLASSES_2021_version = ('airplane', 'bear', 'bird', 'boat', 'car',
                                    'cat', 'cow', 'deer', 'dog', 'duck',
                                    'earless_seal', 'elephant', 'fish',
                                    'flying_disc', 'fox', 'frog', 'giant_panda',
                                    'giraffe', 'horse', 'leopard', 'lizard',
                                    'monkey', 'motorbike', 'mouse', 'parrot',
                                    'person', 'rabbit', 'shark', 'skateboard',
                                    'snake', 'snowboard', 'squirrel', 'surfboard',
                                    'tennis_racket', 'tiger', 'train', 'truck',
                                    'turtle', 'whale', 'zebra', 'dummy')

            cls.METAINFO = dict(classes=CLASSES_2021_version)
        elif dataset_version == 'nus':
            CLASSES_2021_version = ('car', 'truck', 'trailer', 'bus', 'construction_vehicle',
                  'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
                  'barrier')

            cls.METAINFO = dict(classes=CLASSES_2021_version)
        elif dataset_version == 'bdd':
            CLASSES_2021_version = ('pedestrian', 'rider', 'car', 'truck', 'bus', 'train',
               'motorcycle', 'bicycle')

            cls.METAINFO = dict(classes=CLASSES_2021_version)
        elif dataset_version == 'mot17':
            CLASSES_2021_version = ('pedestrian', )# 'person_on_vehicle', 'car', 'bicycle', 'motorbike', 'non_mot_vehicle',
            #    'static_person', 'distractor', 'occluder', 'occluder_on_ground', 'occluder_full', 'reflection', 'crowd')

            cls.METAINFO = dict(classes=CLASSES_2021_version)
        elif dataset_version == 'crowdhuman':
            CLASSES_2021_version = ('pedestrian', )

            cls.METAINFO = dict(classes=CLASSES_2021_version)
        else:
            CLASSES_TN = (0, )

            cls.METAINFO = dict(classes=CLASSES_TN)

    @staticmethod
    def __getname__(): 
        return "youtubevis"
    
    def get_version(self): 
        return self.dataset_version

    def __getitem__(self, idx: int) -> dict:
        """Get the idx-th image and data information of dataset after
        ``self.pipeline``, and ``full_init`` will be called if the dataset has
        not been fully initialized.

        During training phase, if ``self.pipeline`` get ``None``,
        ``self._rand_another`` will be called until a valid image is fetched or
         the maximum limit of refetech is reached.

        Args:
            idx (int): The index of self.data_list.

        Returns:
            dict: The idx-th image and data information of dataset after
            ``self.pipeline``.
        """
        # Performing full initialization by calling `__getitem__` will consume
        # extra memory. If a dataset is not fully initialized by setting
        # `lazy_init=True` and then fed into the dataloader. Different workers
        # will simultaneously read and parse the annotation. It will cost more
        # time and memory, although this may work. Therefore, it is recommended
        # to manually call `full_init` before dataset fed into dataloader to
        # ensure all workers use shared RAM from master process.
        if not self._fully_initialized:
            print_log(
                'Please call `full_init()` method manually to accelerate '
                'the speed.',
                logger='current',
                level=logging.WARNING)
            self.full_init()

        if self.test_mode:
            data = self.prepare_data(idx)
            if data is None:
                raise Exception('Test time pipline should not get `None` '
                                'data_sample')
            return data

        for _ in range(self.max_refetch + 1):
            try:
                data = self.prepare_data(idx)
                # Broken images or random augmentations may cause the returned data
                # to be None
                bboxes = data['data_samples'].gt_instances.bboxes
                bwidths = bboxes[:, 2] - bboxes[:, 0]  # x2 - x1
                bheights = bboxes[:, 3] - bboxes[:, 1]  # y2 - y1
                has_zero_width_or_height = torch.any((bwidths < 1e-2) | (bheights < 1e-2))

                if data is None or has_zero_width_or_height.item():
                    idx = self._rand_another()
                    continue
            except Exception as e:
                # Log the error if needed
                print(f"Error while preparing data for idx {idx}: {e}")
                idx = self._rand_another()
                continue

            # Convert BGR to RGB
            ori_image = data['inputs']['img'][:, [2, 1, 0], :, :]
            # Normalize the image
            # ori_image = ori_image / 127.5 - 1

            _, _, height, width = ori_image.shape
            bboxes = data['data_samples'].gt_instances.bboxes
            classes_name_idx = data['data_samples'].gt_instances.labels
            ins_id = data['data_samples'].gt_instances.instances_id
            map_instances_to_img_idx = data['data_samples'].gt_instances.map_instances_to_img_idx
            if self.dataset_version != 'crowdhuman':
                frame_ids = data['data_samples'].frame_id
                sorted_indices = [_i[0] for _i in sorted(enumerate(frame_ids), key=lambda x: x[1])]
                assert max(frame_ids) - min(frame_ids) == (len(frame_ids) - 1) * self.interval
            else:
                sorted_indices = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0]

            bboxes[:, [0, 2]] /= width   # x_min and x_max
            bboxes[:, [1, 3]] /= height  # y_min and y_max
            # padding none
            nframes = 14
            fixed_objs = 20
            max_length = nframes * fixed_objs
            num_objects = bboxes.shape[0]
            # 对bboxes进行padding
            if len(bboxes) < max_length:
                padding_size = max_length - num_objects
                padding_tensor = torch.full((padding_size, bboxes.shape[1]), float(-1))  # 假设bboxes是二维的
                if num_objects != 0:
                    bboxes_padded = torch.cat([bboxes, padding_tensor], dim=0)
                else:
                    bboxes_padded = padding_tensor
            else:
                bboxes_padded = bboxes[:max_length]

            if num_objects < max_length:
                padding_size = max_length - num_objects
                padding_tensor = torch.full((padding_size,), float(-1))
                if num_objects != 0:
                    classes_name_idx = torch.cat([classes_name_idx, padding_tensor], dim=0)
                else:
                    classes_name_idx = padding_tensor
            else:
                classes_name_idx = classes_name_idx[:max_length]
            
            classes_name_idx = classes_name_idx.int()

            unique_ins_ids = torch.unique(ins_id)
            id_to_index_map = {int(id): index for index, id in enumerate(unique_ins_ids)}

            padding_bboxes = torch.zeros(nframes, fixed_objs, 4)
            padding_phrase = torch.zeros(nframes, fixed_objs)
            padding_ins_id = torch.full((nframes, fixed_objs), -1)
            mask = torch.zeros(nframes, fixed_objs)
            for idx, bbox in enumerate(bboxes_padded[:num_objects]):
                instance_id = ins_id[idx]
                frame_idx = map_instances_to_img_idx[idx]
                
                insertion_idx = id_to_index_map.get(int(instance_id))
                if insertion_idx is None or insertion_idx >= fixed_objs:
                    continue

                padding_bboxes[frame_idx, insertion_idx] = bbox
                padding_ins_id[frame_idx, insertion_idx] = instance_id
                padding_phrase[frame_idx, insertion_idx] = classes_name_idx[idx] if self.dataset_version != 'trackingnet' else 99
                mask[frame_idx, insertion_idx] = 1

            padding_bboxes = padding_bboxes[sorted_indices]
            padding_phrase = padding_phrase[sorted_indices]
            padding_ins_id = padding_ins_id[sorted_indices]
            ori_image = ori_image[sorted_indices]
            mask = mask[sorted_indices]
            
            ori_image_backup = ori_image / 127.5 - 1

            # dummy img
            # ori_image = torch.zeros_like(ori_image)
            color_tensor = torch.tensor(color_list,)
            color_tensor = color_tensor[torch.randperm(color_tensor.size(0))]
            drawn_imgs = draw_bboxes_on_image(ori_image, padding_bboxes, (height, width), color_tensor.tolist())
            color_tensor = color_tensor.to(torch.float32) / 255.0
            ori_image = [to_tensor(img) for img in drawn_imgs]
            ori_image = torch.stack(ori_image) * 2.0 - 1.0

            padding_bboxes_with_color = torch.zeros(padding_bboxes.shape[0], padding_bboxes.shape[1], 7)
            padding_bboxes_with_color[:, :, :4] = padding_bboxes
            padding_bboxes_with_color[:, :, 4:] = color_tensor[:fixed_objs, :]
            # get caption
            text_prompt = 'A segment of multi-object tracking video.'
            if self.get_version() == 'lasot':
                video_key = data['data_samples'].img_path[0].split('/')[2]
                video_path = os.path.dirname(data['data_samples'].img_path[0]).replace('img', 'nlp.txt')
                text_prompt = mmengine.list_from_file(video_path)[0]
            elif self.get_version() == 'nus':
                video_key = data['data_samples'].img_path[0].split('/')[-2]
                text_prompt = 'camera front'
            else:
                video_key = data['data_samples'].img_path[0].split('/')[-2]
            if self.caption_data is not None:
                text_prompt = self.caption_data.get(video_key, "")
            return {
                "ori_pixel": ori_image_backup,
                "pixel_values": ori_image,
                "text_prompt": text_prompt,
                "seg_phrase": padding_phrase,
                'bboxes': padding_bboxes_with_color,
                'instances_id': padding_ins_id,
                'num_objects': num_objects,
                'mask': mask,
                'video_name': video_key,
                'dataset': self.get_version()
            }

        raise Exception(f'Cannot find valid image after {self.max_refetch}! '
                        'Please check your image path and pipeline')
    
    def ref_img_sampling(self,
                         idx: int,
                         data_info: dict,
                         frame_range: list,
                         stride: int = 1,
                         num_ref_imgs: int = 1,
                         filter_key_img: bool = True,
                         method: str = 'uniform') -> List[dict]:
        """Sampling reference frames in the same video for key frame.

        Args:
            idx (int): The index of `data_info`.
            data_info (dict): The information of key frame.
            frame_range (List(int) | int): The sampling range of reference
                frames in the same video for key frame.
            stride (int): The sampling frame stride when sampling reference
                images. Default: 1.
            num_ref_imgs (int): The number of sampled reference images.
                Default: 1.
            filter_key_img (bool): If False, the key image will be in the
                sampling reference candidates, otherwise, it is exclude.
                Default: True.
            method (str): The sampling method. Options are 'uniform',
                'bilateral_uniform', 'test_with_adaptive_stride',
                'test_with_fix_stride'. 'uniform' denotes reference images are
                randomly sampled from the nearby frames of key frame.
                'bilateral_uniform' denotes reference images are randomly
                sampled from the two sides of the nearby frames of key frame.
                'test_with_adaptive_stride' is only used in testing, and
                denotes the sampling frame stride is equal to (video length /
                the number of reference images). test_with_fix_stride is only
                used in testing with sampling frame stride equalling to
                `stride`. Default: 'uniform'.

        Returns:
            list[dict]: `data_info` and the reference images information.
        """
        assert isinstance(data_info, dict)
        if isinstance(frame_range, int):
            assert frame_range >= 0, 'frame_range can not be a negative value.'
            frame_range = [-frame_range, frame_range]
        elif isinstance(frame_range, list):
            assert len(frame_range) == 2, 'The length must be 2.'
            assert frame_range[0] <= 0 and frame_range[1] >= 0
            for i in frame_range:
                assert isinstance(i, int), 'Each element must be int.'
        else:
            raise TypeError('The type of frame_range must be int or list.')

        if 'test' in method and \
                (frame_range[1] - frame_range[0]) != num_ref_imgs:
            logger = MMLogger.get_current_instance()
            logger.info(
                'Warning:'
                "frame_range[1] - frame_range[0] isn't equal to num_ref_imgs."
                'Set num_ref_imgs to frame_range[1] - frame_range[0].')
            self.ref_img_sampler[
                'num_ref_imgs'] = frame_range[1] - frame_range[0]

        if (not self.load_as_video) or data_info.get('frame_id', -1) < 0 \
                or (frame_range[0] == 0 and frame_range[1] == 0):
            ref_data_infos = []
            for i in range(num_ref_imgs):
                ref_data_infos.append(data_info.copy())
        else:
            frame_id = data_info['frame_id']
            left = max(0, frame_id + frame_range[0])
            right = min(frame_id + frame_range[1],
                        data_info['video_length'] - 1)
            frame_ids = list(range(0, data_info['video_length']))

            ref_frame_ids = []
            if method == 'uniform':
                valid_ids = frame_ids[left:right + 1]

                if frame_id not in valid_ids:
                    raise ValueError("frame_id not found in valid_ids")

                idx_of_frame_id = valid_ids.index(frame_id)

                # Start by adding frame_id
                selected_ids = []
                
                left_pointer, right_pointer = idx_of_frame_id - self.interval, idx_of_frame_id + self.interval
                while len(selected_ids)+1 < num_ref_imgs:
                    if left_pointer >= 0:  # If there's a frame to the left
                        selected_ids.insert(0, valid_ids[left_pointer])
                        left_pointer -= self.interval
                    elif right_pointer < len(valid_ids):  # If there's a frame to the right
                        selected_ids.append(valid_ids[right_pointer])
                        right_pointer += self.interval
                    else:
                        break  # No more frames to add

                ref_frame_ids.extend(selected_ids)
            elif method == 'bilateral_uniform':
                assert num_ref_imgs % 2 == 0, \
                    'only support load even number of ref_imgs.'
                for mode in ['left', 'right']:
                    if mode == 'left':
                        valid_ids = frame_ids[left:frame_id + 1]
                    else:
                        valid_ids = frame_ids[frame_id:right + 1]
                    if filter_key_img and frame_id in valid_ids:
                        valid_ids.remove(frame_id)
                    num_samples = min(num_ref_imgs // 2, len(valid_ids))
                    sampled_inds = random.sample(valid_ids, num_samples)
                    ref_frame_ids.extend(sampled_inds)
            elif method == 'test_with_adaptive_stride':
                if frame_id == 0:
                    stride = float(len(frame_ids) - 1) / (num_ref_imgs - 1)
                    for i in range(num_ref_imgs):
                        ref_id = round(i * stride)
                        ref_frame_ids.append(frame_ids[ref_id])
            elif method == 'test_with_fix_stride':
                if frame_id == 0:
                    for i in range(frame_range[0], 1):
                        ref_frame_ids.append(frame_ids[0])
                    for i in range(1, frame_range[1] + 1):
                        ref_id = min(round(i * stride), len(frame_ids) - 1)
                        ref_frame_ids.append(frame_ids[ref_id])
                elif frame_id % stride == 0:
                    ref_id = min(
                        round(frame_id + frame_range[1] * stride),
                        len(frame_ids) - 1)
                    ref_frame_ids.append(frame_ids[ref_id])
                data_info['num_left_ref_imgs'] = abs(frame_range[0])
                data_info['frame_stride'] = stride
            else:
                raise NotImplementedError

            ref_data_infos = []
            for ref_frame_id in ref_frame_ids:
                offset = ref_frame_id - frame_id
                ref_data_info = self._get_ori_data_info(
                    self.valid_data_indices[idx] + offset)

                # We need data_info and ref_data_info to have the same keys.
                for key in data_info.keys():
                    if key not in ref_data_info:
                        ref_data_info[key] = data_info[key]

                ref_data_infos.append(ref_data_info)

            ref_data_infos = sorted(
                ref_data_infos, key=lambda i: i['frame_id'])
        return [data_info, *ref_data_infos]
        

if __name__ == "__main__":
    from mmengine.config import Config, ConfigDict
    from mmtrack.registry import DATASETS
    from mmengine import build_from_cfg
    
    cfg = Config.fromfile('youtube_cfg.py')
    train_dataset = build_from_cfg(cfg.train_dataloader.dataset, DATASETS)