import argparse
import json
import os

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from matplotlib import pyplot as plt
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

USE_LLAVA_CAPTIONER = True
CROP_IMAGE = True

def get_square_coordinates(x, y, l, max_axis):
    half_length = l // 2
    min_x = max(0, x - half_length)
    max_x = min(x + half_length, max_axis)
    min_y = max(0, y - half_length)
    max_y = min(y + half_length, max_axis)

    return min_x, min_y, max_x, max_y


def create_prompts_from_coco_ann(args: argparse.Namespace):
    """ Create prompts for images usable for ControlNet~~/LoRA~~ training starting from COCO formatted annotations
    (e.g. captions_train{year}.json and instances_train{year}.json files)
    Note that bbox in COCO format are expressed like [x_min, y_min, width, height]
    """
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
                    im_dict["captions"].append(an.get("caption", ""))
        # Read all instances and link them to each image
        with open(args.instances_file) as f:
            data = json.load(f)
            data_cats = data["categories"]
            cats = {}
            for cat in data_cats:
                if cat["name"] not in ["person"]:  # else reject
                    cats[cat["id"]] = cat["name"]

            insts = data["annotations"]
            print("Reading instances:")
            idx_inst=0
            for inst in tqdm(insts):
                instance_cat_id = inst["category_id"]
                # check allowed category, area size and that bb does not contains many instances
                if instance_cat_id in cats and inst["area"]>=300 and (len(inst["segmentation"])==0 or len(inst["segmentation"])==1):  # else reject
                    try:
                        im_dict = dataset[inst["image_id"]]
                    except:
                        print("Missing segmentation key in dataset", inst["image_id"]); continue
                    try:
                        image = cv2.imread(os.path.join(args.input_path, im_dict["file"]))
                    except:
                        print("ERR: Could not read image", os.path.join(args.input_path, im_dict["file"])); continue

                    # (optional) center crop target
                    if CROP_IMAGE:
                        left, top, right, bottom = get_square_coordinates(im_dict["width"]//2, im_dict["height"]//2, min(im_dict["height"], im_dict["width"]), max(im_dict["height"], im_dict["width"]))
                        image = image[top:bottom, left:right]

                    size_pre_resize = image.shape
                    bbox = np.array(inst["bbox"])
                    # check bounding box of object (used as controlnet conditioning input)
                    if bbox[0] - left>0 and bbox[1] - top>0 and bbox[0] - left + bbox[2]<size_pre_resize[1] and bbox[1] - top + bbox[3]<size_pre_resize[0]:   # else region cropped, reject
                        # get caption
                        if USE_LLAVA_CAPTIONER:
                            query = (f"Provide an OBJECTIVE and DESCRIPTIVE caption for the scene in the image without hallucinating or making hypothesis out of what is visible, start with `A photo of` and make sure to mention"
                            f" the object `{cats[instance_cat_id]}`, and if present the subject of the action performed on this object, if in hand append {cats[instance_cat_id]} in hand."
                            f" Make a sentence with present participle verbs and indeterminate article. Max 250 characters allowed")
                            prompt = llava_pred.predict(os.path.join(args.input_path, im_dict["file"]), query, 0.7, 0.2, 512)
                            prompt = (prompt[:-1] if prompt.endswith('.') else prompt).replace("\"", "").replace("'", "")
                        else:
                            valid_captions = [cats[instance_cat_id] in c.lower() or cats[instance_cat_id].strip() in c.lower() for c in im_dict["captions"]]
                            prompt = im_dict["captions"][np.argmin(valid_captions)] if any(valid_captions) else None
    
                        if prompt is not None:  # else reject
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
                            crop_left, crop_top, crop_right, crop_bottom = get_square_coordinates(x+(w//2), y+(h//2), max(h, w), 512)  # since COCO contains small objects and the VQVAE of SD has a 8x downscaling effect, we crop a square area 
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
                idx_inst +=1


def check_overlap(boxes_list, filter, th=None, iou_th=None): 
    """
    Check area of intersection of `boxes_list` elements respect `filter` elements
    :param boxes_list: list of list of boxes coordinates [[x0, y0, x1, y1], ...]
    :param filter: list of list of boxes coordinates [[x0, y0, x1, y1], ...]
    :return: Boolean list where True corresponds to an overlap of the corresponding element of `boxes_list` with an element of `filter`  
    """
    filter_list = []
    for reference in boxes_list:
        overlap = False
        for candidate in filter:
            # coordinates of the intersection rectangle
            xA = max(candidate[0], reference[0])
            yA = max(candidate[1], reference[1])
            xB = min(candidate[2], reference[2])
            yB = min(candidate[3], reference[3])
            # area of intersection rectangle
            interWidth = max(0, xB - xA)
            interHeight = max(0, yB - yA)
            interArea = interWidth * interHeight
            # area of both the prediction and reference rectangle
            boxAArea = (candidate[2] - candidate[0]) * (candidate[3] - candidate[1])
            boxBArea = (reference[2] - reference[0]) * (reference[3] - reference[1])
            iou = interArea / float(boxAArea + boxBArea - interArea)

            if (iou_th is not None and iou > iou_th) or (th is not None and interArea > th * boxBArea):
                overlap = True
                break
        filter_list.append(overlap)
    return np.array(filter_list)


def create_prompts_from_hoi_ann(args: argparse.Namespace):
    """ Create prompts for images usable for ControlNet/LoRA training starting from HOIA formatted files
    (e.g. {dataset}/annotations/trainval_{dataset}.json). 
    Note that VCOCO and HICO-DET datasets' shipped annotations have to be converted in HOIA format anyway for HOI task.
    Note that bbox in HOIA format are expressed like [x0, y0, x1, y1]
    """
    from llava_predictor import Predictor
    llava_pred = Predictor()
    llava_pred.setup()
    if args.prompts_type == "lora":
        dino_processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
        dino_model = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-base").to("cuda")

    if not os.path.exists(os.path.join(args.output_path, "target")): os.makedirs(os.path.join(args.output_path, "target"))
    if not os.path.exists(os.path.join(args.output_path, "source")): os.makedirs(os.path.join(args.output_path, "source"))
    if not os.path.exists(os.path.join(args.output_path, "source_obj")): os.makedirs(os.path.join(args.output_path, "source_obj"))
    if not os.path.exists(os.path.join(args.output_path, "mask")): os.makedirs(os.path.join(args.output_path, "mask"))
    if not os.path.exists(os.path.join(args.output_path, "mask_obj")): os.makedirs(os.path.join(args.output_path, "mask_obj"))

    if args.dataset == "vcoco":
        from data.vcoco.categories import object_categories, roles
    else:
        from data.hicodet.categories import object_categories, roles

    with open(os.path.join(args.output_path, 'prompt.json'), 'w') as outfile:
        # Read all images
        with open(args.images_file) as f:
            data = json.load(f)

            inst_idx = 0
            for im_dict in tqdm(data):
                if inst_idx>0 or im_dict["file_name"]=="HICO_train2015_00013074.jpg":
                    inst_idx+=1
                else: continue
                # load the image
                image_path = os.path.join(args.input_path, im_dict["file_name"])
                try:
                    image = cv2.imread(image_path)
                    if image is None: raise Exception()
                except:
                    print("ERR: Could not read image", os.path.join(args.input_path, im_dict["file_name"]));
                    continue

                h, w, _, = image.shape
                if CROP_IMAGE:
                    left, top, right, bottom = get_square_coordinates(w // 2, h // 2, min(h, w), max(h, w))
                    image = image[top:bottom, left:right]
                else:
                    left, top, right, bottom = 0, 0, w, h

                boxes_cropped_resized = {}
                size_pre_resize = image.shape
                for i, ann in enumerate(im_dict["annotations"]):
                    # crop box
                    bbox = ann["bbox"]
                    bbox_crop = np.array([np.clip(bbox[0] - left, 0, size_pre_resize[1]),
                                          np.clip(bbox[1] - top, 0, size_pre_resize[0]),
                                          np.clip(bbox[2] - left, 0, size_pre_resize[1]),
                                          np.clip(bbox[3] - top, 0, size_pre_resize[0])])
                    # resize box
                    x1, x2 = [int(value / size_pre_resize[0] * 512) for value in bbox_crop[[0, 2]]]
                    y1, y2 = [int(value / size_pre_resize[1] * 512) for value in bbox_crop[[1, 3]]]

                    boxes_cropped_resized[i] = [x1, y1, x2, y2]

                # group subjects and their bounding boxes by their object_id
                hoi_groups = {}
                for hoi in im_dict["hoi_annotation"]:
                    # filter by category
                    if hoi["object_id"] < 0 or im_dict["annotations"][hoi["object_id"]]['category_id'] == 1 or hoi["subject_id"] < 0 or im_dict["annotations"][hoi["subject_id"]]['category_id'] != 1:
                        continue

                    # filter by box size
                    th = 0.50
                    object = im_dict["annotations"][hoi["object_id"]]
                    bbox_object = np.array(object["bbox"])
                    bbox_object_crop_res = boxes_cropped_resized[hoi["object_id"]]
                    subject = im_dict["annotations"][hoi["subject_id"]]
                    bbox_subject = np.array(subject["bbox"])
                    bbox_subject_crop_res = boxes_cropped_resized[hoi["subject_id"]]
                    if (bbox_object[0] - left <= -((bbox_object[2]-bbox_object[0]) * th) or  # the left cropping cuts more than `th`% of the object box
                        bbox_object[1] - top <= -((bbox_object[3]-bbox_object[1]) * th) or
                        bbox_object[2] - right >= ((bbox_object[2]-bbox_object[0]) * th) or  # the right cropping cuts more than `th`% of the object box
                        bbox_object[2] - bottom >= ((bbox_object[3]-bbox_object[1]) * th) or
                        ((bbox_object_crop_res[2] - bbox_object_crop_res[0])*(bbox_object_crop_res[3] - bbox_object_crop_res[1])) <= 250 or  # the size of the box is not greater than 250 pixels
                        (args.prompts_type == "lora" and  # subject is ignored in ControlNet preprocessing
                        (bbox_subject[0] - left <= -((bbox_subject[2]-bbox_subject[0]) * th) or  # the left cropping cuts more than `th`% of the subject box
                        bbox_subject[1] - top <= -((bbox_subject[3]-bbox_subject[1]) * th) or
                        bbox_subject[2] - right >= ((bbox_subject[2]-bbox_subject[0]) * th) or  # the right cropping cuts more than `th`% of the subject box
                        bbox_subject[2] - bottom >= ((bbox_subject[3]-bbox_subject[1]) * th) or
                        ((bbox_subject_crop_res[2] - bbox_subject_crop_res[0])*(bbox_subject_crop_res[3] - bbox_subject_crop_res[1])) <= 250))):
                        continue

                    # store hoi group: all subjects interacting with an object [instance_id] performing an action [category_id]
                    group_id = f"{hoi['object_id']} {hoi['category_id']}"
                    if group_id not in hoi_groups:
                        hoi_groups[group_id]={}
                        b2i = boxes_cropped_resized.copy()
                        # object info
                        hoi_groups[group_id]['object_bbox'] = bbox_object_crop_res
                        hoi_groups[group_id]['object_cat'] = object['category_id']
                        hoi_groups[group_id]['role_cat'] = hoi['category_id']
                        hoi_groups[group_id]['boxes_to_ignore'] = b2i
                        del hoi_groups[group_id]['boxes_to_ignore'][hoi["object_id"]]
                    # subject info
                    if args.prompts_type == "lora" and hoi["subject_id"] in hoi_groups[group_id]['boxes_to_ignore']:  # apparently im_dict["hoi_annotation"] contains duplicates
                        hoi_groups[group_id]['subject_bboxes'] = hoi_groups[group_id].get('subject_bboxes', [])+[bbox_subject_crop_res]
                        hoi_groups[group_id]['subject_cat'] = subject['category_id']  # always 1:"person"
                        del hoi_groups[group_id]['boxes_to_ignore'][hoi["subject_id"]]

                # for each HOI (group) annotation, save target and masked image
                for i, hoi_group in enumerate(hoi_groups.values()):
                    query = (f"Provide an OBJECTIVE and DESCRIPTIVE caption for the scene depicted in the image including an identifiable description of the action currently performed in the image by {len(hoi_group['subject_bboxes'])} {object_categories[hoi_group['subject_cat']]} without hallucinating, start with `A photo of` and make sure to mention"
                             + (f" the {len(hoi_group['subject_bboxes'])} {object_categories[hoi_group['subject_cat']]} subject of the action, the object of the action `{object_categories[hoi_group['object_cat']]}`, and the action `{roles[hoi_group['role_cat']]}`."
                                if args.prompts_type == "lora" else f" and describe the object `{object_categories[hoi_group['object_cat']]}`, if in hand append `{object_categories[hoi_group['object_cat']]}` in hand."
                             ) + f" Make a sentence with present participle verbs and indeterminate article. Max 250 characters allowed")

                    prompt = llava_pred.predict(image_path, query, 0.7, 0.2, 512)
                    prompt = (prompt[:-1] if prompt.endswith('.') else prompt).replace("\"", "").replace("'", "")

                    # resize and save target
                    target_resized = cv2.resize(image, (512, 512))
                    target_image = target_resized.copy()
                    target_path = os.path.join("target", os.path.basename(im_dict["file_name"]))
                    cv2.imwrite(os.path.join(args.output_path, target_path), target_image)

                    bgt = [hoi_group['object_bbox']] + hoi_group['subject_bboxes']
                    # obj inpainting: create the source and mask
                    x1, y1, x2, y2 = hoi_group['object_bbox']
                    mask_obj = np.zeros(target_image.shape[:2], dtype=np.uint8)
                    mask_obj[y1:y2+1, x1:x2+1] = 255
                    target_image[y1:y2+1, x1:x2+1] = 0
                    mask_obj_path = os.path.join("mask_obj", os.path.splitext(im_dict["file_name"])[0]+"_"+str(i)+os.path.splitext(im_dict["file_name"])[1])
                    cv2.imwrite(os.path.join(args.output_path, mask_obj_path), mask_obj)
                    source_obj_path = os.path.join("source_obj", os.path.splitext(im_dict["file_name"])[0]+"_"+str(i)+os.path.splitext(im_dict["file_name"])[1])
                    cv2.imwrite(os.path.join(args.output_path, source_obj_path), target_image)

                    if args.prompts_type == "lora":
                        # scene inpainting: create source and mask
                        mask = mask_obj
                        for bbox_subj in hoi_group['subject_bboxes']:
                            x1, y1, x2, y2 = bbox_subj
                            mask[y1:y2+1, x1:x2+1] = 255
                            target_image[y1:y2+1, x1:x2+1] = 0  # cumulative with previous object masking
                        mask_path = os.path.join("mask", os.path.splitext(im_dict["file_name"])[0]+"_"+str(i)+os.path.splitext(im_dict["file_name"])[1])
                        cv2.imwrite(os.path.join(args.output_path, mask_path), mask)
                        source_path = os.path.join("source", os.path.splitext(im_dict["file_name"])[0]+"_"+str(i)+os.path.splitext(im_dict["file_name"])[1])
                        cv2.imwrite(os.path.join(args.output_path, source_path), target_image)
                        
                        # b2i list cover two needs:
                        #  - represent the boxes to ignore during DINO prediction on inpainted image (step necessary for creating a GT box)
                        #  - as well as the list of other annotated boxes (from dataset GT) that the model should detect in the inpainted image
        
                        # re-order HOI annotation list and ids pointers to annotations since some boxes were dropped in the process 
                        new_id = 0
                        b2i_ordered = {}
                        for k,v in hoi_group['boxes_to_ignore'].items():
                            if ((v[2] - v[0])*(v[3] - v[1])) > 100 and not any(check_overlap([v], bgt, th=0.40)):  # check that other GT boxes (excluded the current HOI boxes) hava area non-negative and not lie inside the current HOI boxes 
                                b2i_ordered[k] = {"new_id": new_id, "bbox": v, "category_id": im_dict["annotations"][k]["category_id"]}
                                new_id+=1
                        gt_hoi_annotations = []
                        for ann in im_dict["hoi_annotation"]:
                            if (ann["subject_id"] in b2i_ordered) and (ann["object_id"] in b2i_ordered):
                                gt = ann.copy()
                                gt.update({"subject_id": b2i_ordered[ann["subject_id"]]["new_id"], "object_id": b2i_ordered[ann["object_id"]]["new_id"]})
                                gt_hoi_annotations.append(gt)
                        gt_annotations = [{"bbox": ann["bbox"], "category_id":ann["category_id"]} for ann in b2i_ordered.values()]

                        # DINO detection to extend b2i list
                        text_prompt = f"{object_categories[hoi_group['subject_cat']]}. {object_categories[hoi_group['object_cat']]}."
                        inputs = dino_processor(images=[Image.fromarray(cv2.cvtColor(target_resized, cv2.COLOR_BGR2RGB))], text=[text_prompt], return_tensors="pt", padding=True).to("cuda")
                        outputs = dino_model(**inputs)
                        results = dino_processor.post_process_grounded_object_detection(outputs, inputs.input_ids, box_threshold=0.49, text_threshold=0.3, target_sizes=[(512, 512)])
                        dino_b2i = results[0]["boxes"].cpu()[np.logical_and(~check_overlap(results[0]["boxes"].cpu(), bgt, iou_th=0.70),  [l in [object_categories[hoi_group['subject_cat']], object_categories[hoi_group['object_cat']]] for l in results[0]["labels"]])].tolist() if len(results[0]["boxes"]) > 0 else []

                    # append info
                    if args.prompts_type == "lora":
                        line = {"conditioning": mask_obj_path, "mask": mask_path, "target": target_path,
                                "prompt": prompt, "obj_text": object_categories[object['category_id']],
                                "subject_category": hoi_group['subject_cat'], "object_category": hoi_group['object_cat'], "role_category": hoi_group['role_cat'],
                                "boxes_to_ignore": list(hoi_group['boxes_to_ignore'].values())+dino_b2i, "hoi_annotation": gt_hoi_annotations, "annotations": gt_annotations,
                                }
                    else:
                        line = {"conditioning": mask_obj_path, "mask": mask_obj_path, "target": target_path,
                                "prompt": prompt, "obj_text": object_categories[object['category_id']],
                                }
                    json.dump(line, outfile)
                    outfile.write('\n')
                inst_idx += 1


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["vcoco", "coco", "hicodet"], default="coco", help="Dataset to caption.")
    ap.add_argument("--prompts_type", choices=["ctrl", "lora"], default="ctrl", help="Caption type depending the type of training you aim to do: LoRA -> scene/action inpainting; ControlNet -> object inpainting.")
    ap.add_argument("--images_file", default="captions_train2017.json", help="Filepath to json with images of dataset. (Default COCO captions train file).")
    ap.add_argument("--instances_file", default="instances_train2017.json", help="Filepath to json with instances relative to images of dataset to process. (Default COCO captions train file).")
    ap.add_argument("--input_path", default="./data/coco", help="Path with dataset to process.")
    ap.add_argument("--output_path", default=None, help="Path to save the dataset processed.")

    args = ap.parse_args()
    with (torch.no_grad()):
        if args.prompts_type=="ctrl":
            if args.output_path is None:
                args.output_path = args.dataset+"_formatted"
            if args.dataset=="coco":
                create_prompts_from_coco_ann(args)
            elif args.dataset=="vcoco":
                # In theory, you can use `could create_prompts_from_hoi_ann` to extract prompts from VCOCO, but we use COCO being a superset of it.
                raise Exception("Type of processing not implemented.")
            elif args.dataset=="hicodet":
                create_prompts_from_hoi_ann(args)
        elif args.prompts_type=="lora":
            if args.output_path is None:
                args.output_path = args.dataset + "_formatted2"
            if args.dataset=="coco":
                raise Exception("Type of processing not implemented.")
            else:  # vcoco, hicodet
                create_prompts_from_hoi_ann(args)
