import json
import os
import random

import cv2
import numpy as np
import torch
import transformers
from accelerate.logging import get_logger
from diffusers.image_processor import VaeImageProcessor
# from datasets import load_dataset

from torch.utils.data import Dataset
from torchvision.transforms import transforms

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
        column_names = ["control", "mask", "target", "prompt"]

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

        if args.caption_column is None:
            self.caption_column = "prompt"
            logger.info(f"caption column defaulting to {self.caption_column}")
        else:
            self.caption_column = args.caption_column
            if self.caption_column not in column_names:
                raise ValueError(
                    f"`--caption_column` value '{args.caption_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
                )

        if args.conditioning_image_column is None:
            self.conditioning_image_column = "control"
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
        if random.random() < self.args.proportion_empty_prompts:
            prompt = ""
        elif isinstance(sample_prompt, str):
            prompt = sample_prompt
        elif isinstance(sample_prompt, (list, np.ndarray)):
            # take a random caption if there are multiple
            prompt = random.choice(sample_prompt) if is_train else sample_prompt[0]
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
        # item2 = self.data2[idx]
        target_filename = item[self.image_column]
        mask_filename = item[self.mask_column]
        control_filename = item[self.conditioning_image_column]
        prompt = item[self.caption_column]
        # prompt, neg_prompt = item[self.caption_column], item2[self.caption_column]
        neg_prompt = ""

        try:
            target_image = cv2.imread(os.path.join(self.data_dir, target_filename))
            mask_image = cv2.imread(os.path.join(self.data_dir, mask_filename))
            control_image = cv2.imread(os.path.join(self.data_dir, control_filename))
    
            # Do not forget that OpenCV read images in BGR order.
            target_image = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)
            mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2RGB)
            control_image = cv2.cvtColor(control_image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(os.path.join(self.data_dir, target_filename), os.path.join(self.data_dir, mask_filename), os.path.join(self.data_dir, control_filename), str(e))

        # Preprocessing (color,dims,dtype,resize) and normalize: control images ∈ [0, 1] and images to encode ∈ [-1, 1]
        target = self.image_processor.preprocess(target_image/255., height=self.args.resolution, width=self.args.resolution).squeeze(0)
        mask = self.mask_processor.preprocess(mask_image[:,:,0]/255., height=self.args.resolution, width=self.args.resolution).squeeze(0)
        control = self.control_image_processor.preprocess(mask_image/255., height=self.args.resolution, width=self.args.resolution).squeeze(0)

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