import argparse
import json
import os

import cv2
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt


USE_LLAVA_CAPTIONER = True
CROP_IMAGE = True

def get_square_coordinates(x, y, l, max_axis):
    half_length = l // 2
    min_x = max(0, x - half_length)
    max_x = min(x + half_length, max_axis)
    min_y = max(0, y - half_length)
    max_y = min(y + half_length, max_axis)

    return min_x, min_y, max_x, max_y


def run_data_filter(args: argparse.Namespace):
    if USE_LLAVA_CAPTIONER:
        from llava_predictor import Predictor
        llava_pred = Predictor()
        llava_pred.setup()

    if not os.path.exists(os.path.join(args.output_path, "target")): os.makedirs(os.path.join(args.output_path, "target"))
    if not os.path.exists(os.path.join(args.output_path, "source")): os.makedirs(os.path.join(args.output_path, "source"))
    if not os.path.exists(os.path.join(args.output_path, "mask")): os.makedirs(os.path.join(args.output_path, "mask"))
    if not os.path.exists(os.path.join(args.output_path, "object")): os.makedirs(os.path.join(args.output_path, "object"))
    dataset = {}
    # Read all captions and link them to a single image
    with open(os.path.join(args.output_path, 'prompt.json'), 'w+') as outfile:
        with open(args.images_file) as f:
            data = json.load(f)
            ims = data["images"]
            print("Reading images:")
            for im in tqdm(ims):
                dataset[im["id"]] = {"file": im["file_name"].split("_")[-1], "height": im["height"], "width": im["width"], "captions": []}
            if "annotations" in data:
                ans = data["annotations"]
                print("Reading captions:")
                for an in tqdm(ans):
                    im_dict = dataset[an["image_id"]]
                    im_dict["captions"].append(an["caption"])
        # Read all instances and link them to each image
        with open(args.instances_file) as f:
            data = json.load(f)
            data_cats = data["categories"]
            cats = {}
            for cat in data_cats:
                if "supercategory" not in cat or ("supercategory" in cat and cat["supercategory"] in ["food", "kitchen"] or cat["name"] == "cell phone"):  # else reject
                    cats[cat["id"]] = cat["name"]

            insts = data["annotations"]
            print("Reading instances:")
            for inst in tqdm(insts):
                instance_cat_id = inst["category_id"]
                if instance_cat_id in cats and inst["area"]>=300 and (len(inst["segmentation"])==0 or len(inst["segmentation"])==1):  # else the bb includes many instances, reject
                    try:
                        im_dict = dataset[inst["image_id"]]
                    except:
                        print("Missing segmentation key in dataset", inst["image_id"]); continue

                    if USE_LLAVA_CAPTIONER:
                        query = f"Provide a COMPACT and OBJECTIVE caption to describe the image without hallucinating or making hypothesis out what is visible, focus on {cats[instance_cat_id]} object, use noun phrase with present participle verbs and indeterminate article. Maximum 180 characters allowed."
                        prompt = llava_pred.predict(os.path.join(args.input_path, im_dict["file"]), query, 0.7, 0.2, 512)
                        prompt = (prompt[:-1] if prompt.endswith('.') else prompt).replace("\"", "").replace("'", "")
                    else:
                        valid_captions = [cats[instance_cat_id] in c.lower() or cats[instance_cat_id].strip() in c.lower() for c in im_dict["captions"]]
                        prompt = im_dict["captions"][np.argmin(valid_captions)] if any(valid_captions) else None
                    
                    if prompt is not None:  # else reject
                        try:
                            image = cv2.imread(os.path.join(args.input_path, im_dict["file"]))
                        except:
                            print("ERR: Could not read image", os.path.join(args.input_path, im_dict["file"])); continue

                        # (optional) center crop target
                        if CROP_IMAGE:
                            left, top, right, bottom = get_square_coordinates(im_dict["width"]//2, im_dict["height"]//2, min(im_dict["height"], im_dict["width"]), max(im_dict["height"], im_dict["width"]))
                            image = image[top:bottom, left:right]
                        size_pre_resize = image.shape
                        # check bounding box of object (used as controlnet conditioning input)
                        bbox = np.array(inst["bbox"])

                        if bbox[0] - left>0 and bbox[1] - top>0 and bbox[0] - left+10<size_pre_resize[0] and bbox[1] - top+10<size_pre_resize[1]:   # else region cropped, reject
                            # resize and save target
                            image = cv2.resize(image, (512, 512))
                            target_image = os.path.join("target", os.path.basename(im_dict["file"]))
                            cv2.imwrite(os.path.join(args.output_path, target_image), image)
                            # clip the bounding box coordinates due to crop
                            bbox[:2] = bbox[0] - left, bbox[1] - top
                            bbox[2] = np.clip(bbox[0]+bbox[2], 0, size_pre_resize[0])-bbox[0]
                            bbox[3] = np.clip(bbox[1]+bbox[3], 0, size_pre_resize[1])-bbox[1]
                            # scale the bounding box coordinates due to resize
                            x, w = [int(value / size_pre_resize[0] * 512) for value in bbox[[0, 2]]]
                            y, h = [int(value / size_pre_resize[1] * 512) for value in bbox[[1, 3]]]
                            # crop, scale and save object
                            crop_left, crop_top, crop_right, crop_bottom = get_square_coordinates(x+(w//2), y+(h//2), max(h, w), 512)
                            cropped_image = image[crop_top:crop_bottom, crop_left:crop_right]
                            normalized_image = cv2.resize(cropped_image, (512, 512))
                            object_path = os.path.join("object", os.path.basename(im_dict["file"]))
                            cv2.imwrite(os.path.join(args.output_path, object_path), normalized_image)
                            # mask and source
                            mask = np.zeros(image.shape[:2], dtype=np.uint8)
                            mask[crop_top:crop_bottom, crop_left:crop_right] = 255
                            image[crop_top:crop_bottom, crop_left:crop_right] = 0
                            mask_path = os.path.join("mask", os.path.basename(im_dict["file"]))
                            cv2.imwrite(os.path.join(args.output_path, mask_path), mask)
                            source_path = os.path.join("source", os.path.basename(im_dict["file"]))
                            cv2.imwrite(os.path.join(args.output_path, source_path), image)

                            # append info
                            line = {"conditioning":mask_path, "mask":mask_path,
                                    "target":target_image, "prompt":prompt,
                                    "obj_text":cats[instance_cat_id], "obj_image":object_path}
                            json.dump(line, outfile)
                            outfile.write('\n')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_file", help="Filepath to json with images of dataset.")
    ap.add_argument("--instances_file", help="Filepath to json with instances relative to images of dataset to process.")
    ap.add_argument("--input_path", help="Path with dataset to process.")
    ap.add_argument("--output_path", help="Path to save the dataset processed.")
    # args = vars(ap.parse_args())
    args = ap.parse_args()

    run_data_filter(args)
