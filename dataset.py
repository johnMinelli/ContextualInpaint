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
    def __init__(self, data_dir, resolution=512, tokenizer=None, apply_transformations=False, dilated_conditioning_mask=False):
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
        with open(os.path.join(self.data_dir, 'prompt.json'), 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))

        self.image_column = "target"
        self.prompt_column = "prompt"
        self.conditioning_image_column = "mask"
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
        class_id = torch.tensor([1 if item.get("phone_class",0)==1 else 2 if item.get("cigarette_class",0)==1 else 3 if item.get("food_class",0)==1 else 0])
        prompt_class = "" if item.get("phone_class",0)+item.get("cigarette_class",0)+item.get("food_class",0)==0 else \
            ("Image of a person with " + ("phone" if item.get("phone_class",0)==1 else "cigarette" if item.get("cigarette_class",0)==1 else "food" if item.get("food_class",0)==1 else "nothing") + " in the hands, ")

        try:
            target_image = cv2.cvtColor(cv2.imread(os.path.join(self.data_dir, target_filename)), cv2.COLOR_BGR2RGB)
            mask_image = cv2.imread(os.path.join(self.data_dir, mask_filename), cv2.IMREAD_GRAYSCALE)
            mask_image = mask_augmentation(mask_image, expansion_p=1., patch_p=1., min_expansion_factor=1.1, max_expansion_factor=1.5, patches=2)
            obj_mask_image = cv2.imread(os.path.join(self.data_dir, obj_mask_filename), cv2.IMREAD_GRAYSCALE) if obj_mask_filename is not None else None
            # we use the mask as conditioning for the controlnet
            if self.dilated_conditioning_mask:
                conditioning_image = cv2.cvtColor(mask_image, cv2.COLOR_GRAY2RGB)
            else:
                conditioning_image = cv2.cvtColor(cv2.imread(os.path.join(self.data_dir, mask_filename)), cv2.COLOR_BGR2RGB)

        except Exception as e:
            print("ERR:", os.path.join(self.data_dir, target_filename), os.path.join(self.data_dir, mask_filename), os.path.join(self.data_dir, conditioning_filename), str(e))

        # Preprocessing (color,dims,dtype,resize) and normalize: control images ∈ [0, 1] and images to encode ∈ [-1, 1]
        target = self.image_processor.preprocess(target_image/255., height=self.resolution, width=self.resolution).squeeze(0)
        mask = self.mask_processor.preprocess(mask_image/255., height=self.resolution, width=self.resolution).squeeze(0)
        obj_mask = self.mask_processor.preprocess(obj_mask_image/255., height=self.resolution, width=self.resolution).squeeze(0) if obj_mask_image is not None else None
        conditioning = self.conditioning_image_processor.preprocess(conditioning_image/255., height=self.resolution, width=self.resolution).squeeze(0)

        if self.apply_transformations:
            if torch.rand(1) < 0.2:
                target = self.tr_g(target).repeat(3,1,1)
            if torch.rand(1) < 0.5:
                target = self.tr_f(target)
                mask = self.tr_f(mask)
                conditioning = self.tr_f(conditioning)

            # object_downscaling
            mask_array = mask.squeeze(0).numpy()
            object_area = np.sum(mask_array == 1)
            total_area = mask_array.size
            object_area_ratio = object_area / total_area
            # Calculate the padding size based on the scale factor
            if object_area_ratio > 0.1:
                scale_factor = np.sqrt(0.05 / object_area_ratio)
                padding_width = int(target.size(2) * (1 - scale_factor) / 2)
                padding_height = int(target.size(1) * (1 - scale_factor) / 2)
                # Apply padding to the target image
                padding_transform = transforms.Pad((padding_width, padding_height), fill=0)
                target = padding_transform(target)
                mask = padding_transform(mask)
                conditioning = padding_transform(conditioning)
                # Resize the images back to the original dimensions
                resize_transform = transforms.Resize((target_image.shape[0], target_image.shape[1]))
                target = resize_transform(target)
                mask = resize_transform(mask)
                conditioning = resize_transform(conditioning)

        return {"image": target, "txt": prompt_class+prompt_image, "no_txt": "", "mask": mask, **({"obj_mask": obj_mask} if obj_mask is not None else {}),
                "ctrl_txt": obj_text, "ctrl_txt_image": obj_image,  "conditioning": conditioning,
                "class": class_id}


class ProcDataset(Dataset):
    """ Dataset for generation pipeline. """
    def __init__(self, data_file, num_image, cat_proportions, num_controlnets=0):
        self.data_file = data_file
        self.num_images = num_image
        self.categories = cat_proportions
        self.vae_scale_factor = 8  # rescale the image to be a multiple of: 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.num_controlnets = num_controlnets
        # access json for generation details
        self.generation_template = json.load(codecs.open(self.data_file, 'r', 'utf-8-sig'))
        self.images_rgb = [os.path.join(self.generation_template['images_path_rgb'], f) for f in sorted(os.listdir(self.generation_template['images_path_rgb'])) if not not_image(f)]
        self.images_mask = [os.path.join(self.generation_template['images_path_mask'], f) for f in sorted(os.listdir(self.generation_template['images_path_mask'])) if not not_image(f)]
        self.images_cond = [os.path.join(self.generation_template['images_path_cond'], f) for f in sorted(os.listdir(self.generation_template['images_path_cond'])) if not not_image(f)]
        assert len(self.images_rgb) == len(self.images_mask) == len(self.images_cond), f"The number of images in specified paths must match: RGB {self.images_rgb}, MASK {len(self.images_mask)}, COND {len(self.images_cond)}."

    def __len__(self):
        return self.num_images

    def update_cat_proportions(self, proportions):
        self.categories = proportions

    def generate_prompt(self, json_template):
        category = np.random.choice(list(self.categories.keys()), p=list(self.categories.values()))
        gender = np.random.choice(json_template["gender"])
        pose = np.random.choice(json_template["pose"] + json_template["category"][category]["pose"])
        expression = np.random.choice(json_template["expression"])
        prompts = np.random.choice(json_template["category"][category]["prompts"])

        return {"prompt": json_template["template"].format(gender=gender, action=prompts["action"], pose=pose, expression=expression),
                "focus": prompts["focus"],
                "control": prompts["control"],
                "category": category}

    def __getitem__(self, idx):
        image_id = np.random.randint(len(self.images_rgb))

        generated_prompt = self.generate_prompt(self.generation_template)
        prompt = generated_prompt.get("prompt", None)
        control_prompt = [generated_prompt.get("prompt", None), generated_prompt["control"]] if self.num_controlnets>1 else generated_prompt.get("control", prompt) if self.num_controlnets==1 else None
        focus_prompt = generated_prompt.get("focus", "" if self.num_controlnets>1 else None)
        neg_prompt = self.generation_template.get("neg_prompt", "")
        onehot_class = torch.tensor([1 if generated_prompt["category"]=="phone" else 2 if generated_prompt["category"]=="cigarette" else 3 if generated_prompt["category"] in ["food", "drink"] else 0])

        try:
            source_image = Image.open(self.images_rgb[image_id]).convert("RGB")
            mask_image = Image.open(self.images_mask[image_id]).convert("L")
            mask_image = mask_augmentation(mask_image, expansion_p=1, patch_p=0., min_expansion_factor=1.2, max_expansion_factor=1.5)
            mask_conditioning_image = mask_image.convert("RGB")
            if self.num_controlnets>1:
                mask_conditioning_image = [mask_conditioning_image] * self.num_controlnets
        except Exception as e:
            print("ERR:", self.images_rgb[image_id], self.images_mask[image_id], self.images_cond[image_id], str(e))

        # Returns PIL Images or Tensors preprocessed (color,dims,dtype,resize) and normalized: control images ∈ [0, 1] and images to encode ∈ [-1, 1]
        source = source_image
        mask = mask_image
        mask_conditioning = mask_conditioning_image

        return {"prompt":prompt, "control_prompt":control_prompt, "neg_prompt":neg_prompt, "focus_prompt":focus_prompt, "image":source, "image_name":self.images_rgb[image_id], "class": onehot_class, "mask":mask, "mask_conditioning":mask_conditioning, "category":generated_prompt["category"]}


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