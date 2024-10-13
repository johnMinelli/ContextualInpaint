import codecs
import json
import os

import cv2
import numpy as np
import torch
from PIL import Image
from accelerate.logging import get_logger
from diffusers.image_processor import VaeImageProcessor
# from datasets import load_dataset

from torch.utils.data import Dataset
from torchvision.transforms import transforms

from utils.data_utils import not_image, mask_augmentation

logger = get_logger(__name__)


class DiffuserDataset(Dataset):
    """ Dataset for training and validation. """
    def __init__(self, data_dir, data_file='prompt.json', resolution=512, tokenizer=None, apply_transformations=False, dilated_conditioning_mask=False):
        self.data_dir = data_dir
        self.resolution = resolution
        self.tokenizer = tokenizer
        self.apply_transformations = apply_transformations
        self.dilated_conditioning_mask = dilated_conditioning_mask
        self.vae_scale_factor = 8  # rescale the image to be a multiple of: 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True)
        self.mask_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor, do_normalize=False, do_binarize=True, do_convert_grayscale=True)
        self.conditioning_image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True, do_normalize=False)

        self.data = []
        with open(os.path.join(self.data_dir, data_file), 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))

        self.image_column = "target"
        self.prompt_column = "prompt"
        self.conditioning_image_column = "conditioning"
        self.mask_column = "mask"
        self.obj_text_column = "obj_text"
        self.obj_image_column = "obj_image"
        self.obj_mask_column = "obj_mask"

        # Preprocessing the datasets.
        self._image_transforms = transforms.Compose(
            [
                transforms.Resize(self.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                # transforms.CenterCrop(args.resolution),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        self._conditioning_image_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(self.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                # transforms.CenterCrop(args.resolution),
            ]
        )
        self.tr_g = transforms.Grayscale()
        self.tr_f = transforms.RandomHorizontalFlip(p=1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        target_filename = item[self.image_column]
        mask_filename = item[self.mask_column]
        conditioning_filename = item[self.conditioning_image_column]
        prompt_image = item[self.prompt_column]
        obj_text = item.get(self.obj_text_column, "")
        obj_image = os.path.join(self.data_dir, item[self.obj_image_column]) if self.obj_image_column in item else ""
        obj_mask_filename = item.get(self.obj_mask_column, None)
        p, c, f =  item.get("phone_class", 0), item.get("cigarette_class", 0), item.get("food_class", 0)
        class_id = torch.tensor([1 if p==1 else 2 if c==1 else 3 if f==1 else 0])
        prompt_class = "" if p+c+f==0 else "Image of a person with " + ("phone" if p==1 else "cigarette" if c==1 else "food" if f==1 else "nothing") + " in the hands, "

        try:  # read the image files
            target_image = cv2.cvtColor(cv2.imread(os.path.join(self.data_dir, target_filename)), cv2.COLOR_BGR2RGB)
            mask_image = cv2.imread(os.path.join(self.data_dir, mask_filename), cv2.IMREAD_GRAYSCALE)
            mask_image = mask_augmentation(mask_image, expansion_p=1., patch_p=0., min_expansion_factor=1.1, max_expansion_factor=1.5, patches=2)
            obj_mask_image = cv2.imread(os.path.join(self.data_dir, obj_mask_filename), cv2.IMREAD_GRAYSCALE) if obj_mask_filename is not None else None
            # we use the mask as conditioning for the controlnet
            if self.dilated_conditioning_mask:
                conditioning_image = cv2.cvtColor(mask_image, cv2.COLOR_GRAY2RGB)
            else:
                conditioning_image = cv2.cvtColor(cv2.imread(os.path.join(self.data_dir, conditioning_filename)), cv2.COLOR_BGR2RGB)

        except Exception as e:
            print("ERR:", os.path.join(self.data_dir, target_filename), os.path.join(self.data_dir, mask_filename), os.path.join(self.data_dir, conditioning_filename), str(e))

        # Preprocessing (color,dims,dtype,resize) and normalize: control images ∈ [0, 1] and images to encode ∈ [-1, 1]
        target = self.image_processor.preprocess(target_image/255., height=self.resolution, width=self.resolution).squeeze(0)
        mask = self.mask_processor.preprocess(mask_image/255., height=self.resolution, width=self.resolution).squeeze(0)
        obj_mask = self.mask_processor.preprocess(obj_mask_image/255., height=self.resolution, width=self.resolution).squeeze(0) if obj_mask_image is not None else None
        conditioning = self.conditioning_image_processor.preprocess(conditioning_image/255., height=self.resolution, width=self.resolution).squeeze(0)

        if obj_mask is not None:  # for the segmented obj mask we adopt upscale
            h, w = obj_mask.shape[-2:]
            mask_array = obj_mask.squeeze(0).numpy()
            object_area = np.sum(mask_array == 1)
            total_area = mask_array.size
            object_area_ratio = object_area / total_area
            if object_area > 0 and object_area_ratio < 0.001:
                scale_factor = np.sqrt(object_area_ratio/0.001)
                padding_width = int(w * (1 - scale_factor) / 2)//2
                padding_height = int(h * (1 - scale_factor) / 2)//2
                # Apply padding to the target image
                padding_transform = transforms.Pad((padding_width, padding_height), fill=0)
                obj_mask = padding_transform(obj_mask)
                # Resize the images back to the original dimensions
                resize_transform = transforms.Resize((h, w))
                obj_mask = resize_transform(obj_mask)

        if self.apply_transformations:
            if torch.rand(1) < 0.2:
                target = self.tr_g(target).repeat(3,1,1)
            if torch.rand(1) < 0.5:
                target = self.tr_f(target)
                mask = self.tr_f(mask)
                conditioning = self.tr_f(conditioning)

            # object_downscaling
            h, w = mask.shape[-2:]
            mask_array = mask.squeeze(0).numpy()
            object_area = np.sum(mask_array == 1)
            total_area = mask_array.size
            object_area_ratio = object_area / total_area
            # Calculate the padding size based on the scale factor
            if object_area_ratio > 0.1:
                scale_factor = np.sqrt(0.05 / object_area_ratio)
                padding_width = int(w * (1 - scale_factor) / 2)
                padding_height = int(h * (1 - scale_factor) / 2)
                # Apply padding to the target image
                padding_transform = transforms.Pad((padding_width, padding_height), fill=0)
                target = padding_transform(target)
                mask = padding_transform(mask)
                conditioning = padding_transform(conditioning)
                # Resize the images back to the original dimensions
                resize_transform = transforms.Resize((h, w))
                target = resize_transform(target)
                mask = resize_transform(mask)
                conditioning = resize_transform(conditioning)

        return {"image": target, "image_name": target_filename, "txt": prompt_image, "no_txt": "", "mask": mask, **({"obj_mask": obj_mask} if obj_mask is not None else {}),
                "ctrl_txt": obj_text, "ctrl_txt_image": obj_image,  "conditioning": conditioning,
                "class": class_id}


class SynthDataset(Dataset):
    """ Dataset for data synthesis. """

    def __init__(self, data_dir, data_file='prompt.json', num_image=None, cat_probabilities=None, resolution=512, dilated_conditioning_mask=False):
        self.data_dir = data_dir
        self.resolution = resolution
        self.num_images = num_image
        self.categories = cat_probabilities
        self.dilated_conditioning_mask = dilated_conditioning_mask
        self.vae_scale_factor = 8  # rescale the image to be a multiple of: 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True)
        self.mask_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor, do_normalize=False, do_binarize=True, do_convert_grayscale=True)
        self.conditioning_image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True, do_normalize=False)
        self.neg_prompt = "inconsistent styles, blurred, text, watermark, poorly drawn face, poorly drawn hands, disfigured human, deformed body, deformed face, no human, no face, child, baby, many arms, many hands, many legs, unrealistic proportions"
        
        if self.categories is not None:
            # weighted sampling; data indexed by category
            self.data = {}
            with open(os.path.join(self.data_dir, data_file), 'rt') as f:
                for line in f:
                    l = json.loads(line)
                    hoi_category = (int(l["subject_category"]), int(l["role_category"]), int(l["object_category"]))
                    self.data[hoi_category] = self.data.get(hoi_category, []) + [l]
            # filter categories (for sampling) by data categories availability
            self.categories = {k: v for k, v in self.categories.items() if k in self.data}
            # re-normalize category probabilities
            tot_w = sum(self.categories.values())
            if tot_w > 0:
                self.categories = {cat: w / tot_w for cat, w in self.categories.items()}
            else:
                raise Exception("No categories for sampling")
        else:
            # uniform sampling; data is a list of annotations 
            self.data = []
            with open(os.path.join(self.data_dir, data_file), 'rt') as f:
                for line in f:
                    self.data.append(json.loads(line))

        self.image_column = "target"
        self.prompt_column = "prompt"
        self.conditioning_image_column = "conditioning"
        self.mask_column = "mask"
        self.obj_text_column = "obj_text"

        # Preprocessing the datasets.
        self._image_transforms = transforms.Compose(
            [
                transforms.Resize(self.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                # transforms.CenterCrop(args.resolution),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        self._conditioning_image_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(self.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                # transforms.CenterCrop(args.resolution),
            ]
        )

    def __len__(self):
        return self.num_images

    def update_cat_proportions(self, proportions):
        self.categories = proportions

    def __getitem__(self, idx):
        if self.categories is not None:
            hoi_category = eval(np.random.choice([str(k) for k in self.categories.keys()], p=list(self.categories.values())))
            item = np.random.choice(self.data[hoi_category]).copy()
        else:
            item = self.data[idx].copy()
            
        target_filename = item.pop(self.image_column)
        mask_filename = item.pop(self.mask_column)
        conditioning_filename = item.pop(self.conditioning_image_column)
        prompt_image = item.pop(self.prompt_column)
        obj_text = item.pop(self.obj_text_column, "")

        try:  # read the image files
            target_image = cv2.cvtColor(cv2.imread(os.path.join(self.data_dir, target_filename)), cv2.COLOR_BGR2RGB)
            mask_image = cv2.imread(os.path.join(self.data_dir, mask_filename), cv2.IMREAD_GRAYSCALE)
            mask_image = mask_augmentation(mask_image, expansion_p=1., patch_p=0., min_expansion_factor=1.1, max_expansion_factor=1.5, patches=2)
            if self.dilated_conditioning_mask:  # we use the mask as conditioning for the controlnet
                conditioning_image = cv2.cvtColor(mask_image, cv2.COLOR_GRAY2RGB)
            else:
                conditioning_image = cv2.cvtColor(cv2.imread(os.path.join(self.data_dir, conditioning_filename)), cv2.COLOR_BGR2RGB)
        except Exception as e:
            print("ERR:", os.path.join(self.data_dir, target_filename), os.path.join(self.data_dir, mask_filename), os.path.join(self.data_dir, conditioning_filename), str(e))

        # Preprocessing (color,dims,dtype,resize) and normalize: control images ∈ [0, 1] and images to encode ∈ [-1, 1]
        target = self.image_processor.preprocess(target_image/255., height=self.resolution, width=self.resolution).squeeze(0)
        mask = self.mask_processor.preprocess(mask_image/255., height=self.resolution, width=self.resolution).squeeze(0)
        conditioning = self.conditioning_image_processor.preprocess(conditioning_image/255., height=self.resolution, width=self.resolution).squeeze(0)

        return {"image": target, "image_name": target_filename, "txt": prompt_image, "neg_txt": self.neg_prompt, "mask": mask,
                "ctrl_txt": obj_text, "conditioning": conditioning,
                **{k:json.dumps(v) if type(v) in [dict,list] else v for k,v in item.items()}}


if __name__ == '__main__':

    dataset = DiffuserDataset()
    print(len(dataset))

    item = dataset[1234]
    jpg = item['jpg']
    txt = item['txt']
    hint = item['hint']
    print(txt)
    print(jpg.shape)
    print(hint.shape)