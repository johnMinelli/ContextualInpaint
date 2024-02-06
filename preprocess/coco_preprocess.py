import argparse
import json
import os

import cv2
import numpy as np
from tqdm import tqdm

DATA_PATH = "../data/coco_formatted/"

def run_data_filter(args: argparse.Namespace):
    if not os.path.exists(os.path.join(DATA_PATH, "target")): os.makedirs(os.path.join(DATA_PATH, "target"))
    if not os.path.exists(os.path.join(DATA_PATH, "source")): os.makedirs(os.path.join(DATA_PATH, "source"))
    if not os.path.exists(os.path.join(DATA_PATH, "mask")): os.makedirs(os.path.join(DATA_PATH, "mask"))
    if not os.path.exists(os.path.join(DATA_PATH, "object")): os.makedirs(os.path.join(DATA_PATH, "object"))
    dataset = {}
    # Read all captions and link them to a single image
    with open(os.path.join(DATA_PATH, 'prompt.json'), 'w+') as outfile:
        with open(os.path.join(args["input_path"],f"captions_{args['dataset']}.json")) as f:
            data = json.load(f)
            ims = data["images"]
            print("Reading images:")
            for im in tqdm(ims):
                dataset[im["id"]] = {"file": im["file_name"].split("_")[-1], "height": im["height"], "width": im["width"], "captions": []}
            ans = data["annotations"]
            print("Reading captions:")
            for an in tqdm(ans):
                im_dict = dataset[an["image_id"]]
                im_dict["captions"].append(an["caption"])
        # Read all instances and link them to each image
        with open(os.path.join(args["input_path"],f"instances_{args['dataset']}.json")) as f:
            data = json.load(f)
            data_cats = data["categories"]
            cats = {}
            for cat in data_cats:
                # if cat["supercategory"] not in ["appliance", "furniture", "sports", "animal", "outdoor", "vehicle", "person"]:  # else reject
                if cat["supercategory"] in ["food", "kitchen"] or cat["name"] == "cell phone":  # else reject
                    cats[cat["id"]] = cat["name"]

            insts = data["annotations"]
            print("Reading instances:")
            for inst in tqdm(insts):
                instance_cat_id = inst["category_id"]
                if instance_cat_id in cats and len(inst["segmentation"])==1 and inst["area"]>=300:  # else reject
                    try:
                        im_dict = dataset[inst["image_id"]]
                    except:
                        print("Missing segmentation key in dataset", inst["image_id"]); continue

                    valid_captions = [cats[instance_cat_id] in c.lower() or cats[instance_cat_id].strip() in c.lower() for c in im_dict["captions"]]
                    if any(valid_captions):  # else reject
                        prompt = im_dict["captions"][np.argmin(valid_captions)]
                        try:
                            image = cv2.imread(os.path.join(args["input_path"],args['dataset'],im_dict["file"]))
                        except:
                            print("ERR: Could not read image", os.path.join(args["input_path"],args['dataset'],im_dict["file"])); continue

                        # target
                        image = cv2.resize(image, (512, 512))
                        target_image = os.path.join(DATA_PATH, "target", im_dict["file"])
                        cv2.imwrite(target_image, image)
                        # object (used as controlnet cross input)
                        bbox = np.array(inst["bbox"])
                        x, w = [int(value / im_dict["width"] * 512) for value in bbox[[0, 2]]]
                        y, h = [int(value / im_dict["height"] * 512) for value in bbox[[1, 3]]]
                        side_length = max(h, w)
                        if x+side_length > 512: x = 512-side_length
                        if y+side_length > 512: y = 512-side_length
                        cropped_image = image[y:y+side_length, x:x+side_length]
                        normalized_image = cv2.resize(cropped_image, (512, 512))
                        object_path = os.path.join(DATA_PATH, "object", im_dict["file"])
                        cv2.imwrite(object_path, normalized_image)
                        # mask and source
                        mask = np.zeros(image.shape[:2], dtype=np.uint8)
                        mask[y:y+side_length, x:x+side_length] = 255
                        image[y:y+side_length, x:x+side_length] = 0
                        mask_path = os.path.join(DATA_PATH, "mask", im_dict["file"])
                        cv2.imwrite(mask_path, mask)
                        source_path = os.path.join(DATA_PATH, "source", im_dict["file"])
                        cv2.imwrite(source_path, image)

                        # append info
                        line = {"conditioning":mask_path.replace(DATA_PATH,""), "mask":mask_path.replace(DATA_PATH,""),
                                "target":target_image.replace(DATA_PATH,""), "prompt":prompt,
                                "obj_text":cats[instance_cat_id], "obj_image":object_path.replace(DATA_PATH,"")}
                        json.dump(line, outfile)
                        outfile.write('\n')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=['val2014', 'val2017', 'train2014', 'train2017'], help="Choose on which dataset run the preprocessing.")
    ap.add_argument("--input_path", help="Path with dataset to process.")
    args = vars(ap.parse_args())

    run_data_filter(args)
