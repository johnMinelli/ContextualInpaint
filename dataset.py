import json
import os
import random

import cv2
import numpy as np
import torch
import transformers
from PIL import Image
from accelerate.logging import get_logger
from diffusers.image_processor import VaeImageProcessor
# from datasets import load_dataset

from torch.utils.data import Dataset, default_collate
from torchvision.transforms import transforms

from utils.utils import not_image

logger = get_logger(__name__)
class DiffuserDataset(Dataset):
    def __init__(self, args, tokenizer):
        self.args = args
        self.tokenizer = tokenizer
        self.vae_scale_factor = 8  # rescale the image to be a multiple of: 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True)
        self.mask_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor, do_normalize=False, do_binarize=True, do_convert_grayscale=True)
        self.control_image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True, do_normalize=False)
        self.data_dir = args.train_data_dir

        self.data = []
        with open(os.path.join(self.data_dir, 'prompt.json'), 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))

        self.data2 = []
        with open(os.path.join(self.data_dir, 'prompt_short.json'), 'rt') as f:
            for line in f:
                self.data2.append(json.loads(line))

        # # Get the datasets: you can either provide your own training and evaluation files (see below)
        # # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).
        #
        # # In distributed training, the load_dataset function guarantees that only one local process can concurrently
        # # download the dataset.
        # if args.dataset_name is not None:
        #     # Downloading and loading a dataset from the hub.
        #     dataset = load_dataset(
        #         args.dataset_name,
        #         args.dataset_config_name,
        #         cache_dir=args.cache_dir,
        #     )
        # else:
        #     if args.train_data_dir is not None:
        #         dataset = load_dataset(
        #             args.train_data_dir,
        #             cache_dir=args.cache_dir,
        #         )
        #     # See more about loading custom images at
        #     # https://huggingface.co/docs/datasets/v2.0.0/en/dataset_script

        # Preprocessing the datasets.
        column_names = ["conditioning", "mask", "target", "prompt"]

        # 6. Get the column names for input/target.
        if args.image_column is None:
            self.image_column = "target"
            logger.info(f"image column defaulting to {self.image_column}")
        else:
            self.image_column = args.image_column
            if self.image_column not in column_names:
                raise ValueError(
                    f"`--image_column` value '{args.image_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
                )

        if args.prompt_column is None:
            self.prompt_column = "prompt"
            logger.info(f"prompt column defaulting to {self.prompt_column}")
        else:
            self.prompt_column = args.prompt_column
            if self.prompt_column not in column_names:
                raise ValueError(
                    f"`--prompt_column` value '{args.prompt_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
                    )

        if args.conditioning_image_column is None:
            self.conditioning_image_column = "conditioning"
            logger.info(f"conditioning image column defaulting to {self.conditioning_image_column}")
        else:
            self.conditioning_image_column = args.conditioning_image_column
            if self.conditioning_image_column not in column_names:
                raise ValueError(
                    f"`--conditioning_image_column` value '{args.conditioning_image_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
                )

        if args.mask_column is None:
            self.mask_column = "mask"
            logger.info(f"mask column defaulting to {self.mask_column}")
        else:
            self.mask_column = args.mask_column
            if self.mask_column not in column_names:
                raise ValueError(
                    f"`--conditioning_image_column` value '{args.mask_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
                )

        if args.max_train_samples is not None:
            dataset["train"] = dataset["train"].shuffle(seed=args.seed).select(range(args.max_train_samples))

        self._image_transforms = transforms.Compose(
            [
                transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                # transforms.CenterCrop(args.resolution),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        self._conditioning_image_transforms = transforms.Compose(
            [   
                transforms.ToTensor(),
                transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                # transforms.CenterCrop(args.resolution),
            ]
        )

    def _tokenize_prompt(self, sample_prompt, is_train=True):
        if np.random.random() < self.args.proportion_empty_prompts:
            prompt = ""
        elif isinstance(sample_prompt, str):
            prompt = sample_prompt
        elif isinstance(sample_prompt, (list, np.ndarray)):
            # take a random caption if there are multiple
            prompt = np.random.choice(sample_prompt) if is_train else sample_prompt[0]
        else:
            raise ValueError(
                f"The prompt for the sample provided {sample_prompt} should contain either strings or lists of strings."
            )
        inputs = self.tokenizer([prompt], max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt")
        return inputs.input_ids[0]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        target_filename = item[self.image_column]
        mask_filename = item[self.mask_column]
        conditioning_filename = item[self.conditioning_image_column]
        prompt = item[self.prompt_column]
        obj_text = item.get(self.obj_text_column, "")
        obj_image = item.get(self.obj_image_column, "")

        try:
            target_image = cv2.imread(os.path.join(data_dir, target_filename))
            mask_image = cv2.imread(os.path.join(data_dir, mask_filename))
            conditioning_image = cv2.imread(os.path.join(data_dir , conditioning_filename))
    
            # Do not forget that OpenCV read images in BGR order.
            target_image = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)
            mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2RGB)
            conditioning_image = cv2.cvtColor(conditioning_image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print("ERR:", os.path.join(data_dir, target_filename), os.path.join(data_dir, mask_filename), os.path.join(data_dir, conditioning_filename), str(e))

        # Preprocessing (color,dims,dtype,resize) and normalize: control images ∈ [0, 1] and images to encode ∈ [-1, 1]
        target = self.image_processor.preprocess(target_image/255., height=self.args.resolution, width=self.args.resolution).squeeze(0)
        mask = self.mask_processor.preprocess(mask_image[:,:,0]/255., height=self.args.resolution, width=self.args.resolution).squeeze(0)
        conditioning = self.conditioning_image_processor.preprocess(mask_image/255., height=self.args.resolution, width=self.args.resolution).squeeze(0)

        return {"image": target, "txt": prompt, "no_txt": "", "ctrl_txt": obj_text, "ctrl_txt_image": obj_image, "conditioning": conditioning, "mask": mask}


def proc_collate_fn(data):
    # collated_data = default_collate(data)
    collated_data = {k:[b[k] for b in data] for k,v in data[0].items()}
    return collated_data

class ProcDataset(Dataset):
    def __init__(self, args, cat_proportions, num_controlnets=0):
        self.args = args
        self.categories = cat_proportions
        self.num_images = args.num_validation_images
        self.vae_scale_factor = 8  # rescale the image to be a multiple of: 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.num_controlnets = num_controlnets
        # access json for generation details
        self.generation_template = json.load(codecs.open(args.evaluation_file, 'r', 'utf-8-sig'))
        self.images_rgb = [os.path.join(self.generation_template['images_path_rgb'], f) for f in sorted(os.listdir(self.generation_template['images_path_rgb'])) if not not_image(f)]
        self.images_mask = [os.path.join(self.generation_template['images_path_mask'], f) for f in sorted(os.listdir(self.generation_template['images_path_mask'])) if not not_image(f)]
        self.images_cond = [os.path.join(self.generation_template['images_path_cond'], f) for f in sorted(os.listdir(self.generation_template['images_path_cond'])) if not not_image(f)]
        assert len(self.images_rgb) == len(self.images_mask) == len(self.images_cond), f"The number of images in specified paths must match: RGB {self.images_rgb}, MASK {len(self.images_mask)}, COND {len(self.images_cond)}."

        return {"control": control, "mask": mask, "image": target, "txt": prompt, "neg_txt": neg_prompt}


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