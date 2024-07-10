import argparse
import csv
import json
import os
import shutil
import warnings
import requests

import cv2
import matplotlib.pyplot as plt

import numpy as np
import torch
from torchvision.transforms.functional import pil_to_tensor
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from tqdm import tqdm
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

def not_image(filename: str):
    return not filename.lower().endswith(".jpeg") and not filename.lower().endswith(".jpg") and not filename.lower().endswith(".png")

DATA_PATH = "../data/verizon_formatted2/"

def clean_preprocessed_data(args):
    def remove(files):
        for f in files:
            os.remove(f)

    for root, dirs, files in os.walk(os.path.join(args.get("input_path", DATA_PATH), "source")):
        if len(files)==0: continue
        print("Entering folder:", root)
        for filename in tqdm(files):
            if not_image(filename): continue
            rgb_path = os.path.join(root, filename)
            mask_path = os.path.join(root.replace("source", "mask"), filename)
            try:
                rgb_im = cv2.imread(rgb_path)
                mask_im = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                shape = rgb_im.shape
                if rgb_im is None or mask_im is None: raise Exception()
            except Exception as e:
                print("ERR", os.path.join(root, filename), str(e))
                continue
            mask_im = cv2.threshold(mask_im,50,255,cv2.THRESH_BINARY)[1]
            if np.any(mask_im==255):
                # right crop
                rgb_im = rgb_im[:,-shape[0]:]
                mask_im = mask_im[:,-shape[0]:]
                if np.sum(mask_im[:,0]>50)/shape[0] < 0.5 and (np.sum(mask_im[:,:]>50)/(shape[0]*shape[1]))>0.16 and (np.sum(mask_im[:,:]>50)/(shape[0]*shape[1]))<0.85:
                    # resize
                    rgb_im = cv2.resize(rgb_im, (512, 512))
                    mask_im = cv2.resize(mask_im, (512, 512))
                    # save
                    cv2.imwrite(rgb_path, rgb_im)
                    cv2.imwrite(mask_path, mask_im)
                else:
                    remove([rgb_path, mask_path])
            else:
                remove([rgb_path, mask_path])

    root = args.get("input_path", DATA_PATH)
    if os.path.exists(os.path.join(root, "prompt.json")):
        updated_lines = []
        # read lines
        with open(os.path.join(root, "prompt.json"), 'r') as file:
            lines = file.readlines()
        # check file existence for each line 
        for line in lines:
            json_data = json.loads(line)
            if os.path.exists(os.path.join(root, json_data["target"])) and os.path.exists(os.path.join(root, json_data["mask"])):
                updated_lines.append(line)
        # overwrite the json
        with open(os.path.join(root, "prompt.json"), 'w') as file:
            for line in updated_lines:
                file.write(line)


def run_masking(args: argparse.Namespace):
    cfg = get_cfg()
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.4  # set threshold for this model
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    predictor = DefaultPredictor(cfg)
    dil_kernel = np.ones((7, 7), np.uint8)  # Adjust the kernel size according to your requirements
    sam_model_path = "/tmp/sam_vit_h_4b8939.pth"
    if not os.path.exists(sam_model_path):
        print("Download SAM model...")
        r = requests.get('https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth', allow_redirects=True)
        open(sam_model_path, 'wb').write(r.content)
    else: print("Using SAM model cached.")
    sam = sam_model_registry["vit_h"](checkpoint=sam_model_path).cuda()
    mask_predictor = SamPredictor(sam)
    # Create output folder if not exist
    source_folder = os.path.join(args.get("output_path", DATA_PATH), "source")
    mask_folder = os.path.join(args.get("output_path", DATA_PATH), "mask")
    target_folder = os.path.join(args.get("output_path", DATA_PATH), "target")
    if not os.path.exists(source_folder):
        os.makedirs(source_folder)
    if not os.path.exists(mask_folder):
        os.makedirs(mask_folder)
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # Create masked source files
    if os.path.exists(os.path.join(args.get("input_path", DATA_PATH), "target")):
        print("'target' folder will be used as input path.")
        args["input_path"] = os.path.join(args.get("input_path", DATA_PATH), "target")
    for root, dirs, files in os.walk(args.get("input_path", DATA_PATH)):
        if len(files) == 0: continue
        print("Entering folder:", root)
        for filename in tqdm(files):
            if not_image(filename): continue

            try:
                target_im = cv2.cvtColor(cv2.imread(os.path.join(root, filename)), cv2.COLOR_BGR2RGB)
                rgb_im = cv2.cvtColor(cv2.imread(os.path.join(root, filename)), cv2.COLOR_BGR2RGB)
            except Exception as e:
                print("ERR: reading", os.path.join(root, filename), str(e))
                continue
            outputs = predictor(rgb_im)
            person_filter = outputs["instances"].pred_classes==0
            # detectron masks
            pred_masks = outputs["instances"].pred_masks[person_filter]
            # SAM masks
            transformed_boxes = mask_predictor.transform.apply_boxes_torch(outputs["instances"].pred_boxes[person_filter].tensor, rgb_im.shape[:2])
            mask_predictor.set_image(rgb_im)
            try:
                pred_masks = mask_predictor.predict_torch(boxes=transformed_boxes, multimask_output=False, point_coords=None, point_labels=None)[0][:,0]  # masks, scores, logits
            except Exception as e:
                print("ERR: mask detection", os.path.join(root, filename), str(e))
                continue
            # combine masks and dilate
            pred_mask_bool = pred_masks.sum(0).bool().cpu()
            pred_mask_bool = cv2.dilate(pred_mask_bool.numpy().astype(np.uint8) * 255, dil_kernel, iterations=2) > 0

            # v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
            # out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            # plt.imshow(pred_masks)
            # plt.show()

            rgb_im[pred_mask_bool] = [0, 0, 0]
            mask_im = np.zeros(rgb_im.shape[:2])
            mask_im[pred_mask_bool] = 255

            # Save
            plt.imsave(os.path.join(source_folder, filename), rgb_im)
            plt.imsave(os.path.join(mask_folder, filename), mask_im)
            plt.imsave(os.path.join(target_folder, filename), target_im)


def run_pose_estimation(args: argparse.Namespace):
    cfg = get_cfg()  # get a fresh new config
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.65  # set threshold for this model
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml")
    predictor = DefaultPredictor(cfg)

    # Create output folder if not exist
    poses_folder = os.path.join(args.get("output_path", DATA_PATH), "poses")
    if not os.path.exists(poses_folder):
        os.makedirs(poses_folder)

    if os.path.exists(os.path.join(args.get("input_path", DATA_PATH), "target")):
        print("'target' folder will be used as input path.")
        args["input_path"] = os.path.join(args.get("input_path", DATA_PATH), "target")
    for root, dirs, files in os.walk(args.get("input_path", DATA_PATH)):
        if len(files)==0: continue
        print("Entering:", root)
        for filename in tqdm(files):
            if not_image(filename): continue

            try:
                im = cv2.cvtColor(cv2.imread(os.path.join(root, filename)), cv2.COLOR_BGR2RGB)
            except Exception as e:
                print("ERR", os.path.join(root, filename), str(e))
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                outputs = predictor(im)["instances"].to("cpu")

            person_filter = outputs.pred_classes==0
            outputs.pred_boxes = outputs.pred_boxes[person_filter]
            del outputs._fields["pred_boxes"]
            outputs.scores = outputs.scores[person_filter]
            outputs.pred_classes = outputs.pred_classes[person_filter]
            outputs.pred_keypoints = outputs.pred_keypoints[person_filter]
            outputs.pred_keypoint_heatmaps = outputs.pred_keypoint_heatmaps[person_filter]

            v = Visualizer(np.zeros_like(im), MetadataCatalog.get(cfg.DATASETS.TRAIN[0]))
            out = v.draw_instance_predictions(outputs).get_image()[:, :, ::-1]
            # plt.imshow(out)
            # plt.show()            

            # Save
            plt.imsave(os.path.join(poses_folder, filename), out)

def run_object_detection(args: argparse.Namespace):
    import torch
    import csv
    from PIL import Image
    from lang_sam import LangSAM

    def check_overlap(hand_boxes, obj_boxes):
        hand_centers_x = (hand_boxes[:, 0] + hand_boxes[:, 2]) / 2
        hand_centers_y = (hand_boxes[:, 1] + hand_boxes[:, 3]) / 2
        hand_sizes_x = hand_boxes[:, 2] - hand_boxes[:, 0]
        hand_sizes_y = hand_boxes[:, 3] - hand_boxes[:, 1]
        max_distances = (torch.max(hand_sizes_x, hand_sizes_y) / 2) * 2

        obj_centers_x = (obj_boxes[:, 0] + obj_boxes[:, 2]) / 2
        obj_centers_y = (obj_boxes[:, 1] + obj_boxes[:, 3]) / 2
        obj_sizes_x = obj_boxes[:, 2] - obj_boxes[:, 0]
        obj_sizes_y = obj_boxes[:, 3] - obj_boxes[:, 1]
        # max_obj_distances = (torch.max(obj_sizes_x, obj_sizes_y) / 2) + 10

        distances = torch.sqrt((hand_centers_x[:, None] - obj_centers_x[None, :]) ** 2 + (hand_centers_y[:, None] - obj_centers_y[None, :]) ** 2)
        return torch.any(distances < max_distances[:, None], 0)

    model = LangSAM()

    root = args.get("input_path", DATA_PATH)+"/"
    if not os.path.exists(os.path.join(root, "obj_mask")):
        os.makedirs(os.path.join(root, "obj_mask"))

    csv_file = args.get("csv", None)
    label_map = {}
    if csv_file is not None:
        with open(csv_file, "r") as file:
            csv_reader = csv.DictReader(file)
            for row in csv_reader:
                label_id = row["label_id"]
                phone_class = int(row["phone_binary"])
                cigarette_class = int(row["cigarette_binary"])
                food_class = int(row["food_binary"])
                label_map[label_id] = {"phone_class": phone_class, "cigarette_class": cigarette_class, "food_class": food_class}

    with torch.no_grad():
        if os.path.exists(os.path.join(root, "prompt.json")):
            updated_lines = []
            objects_identified = {}
            # read lines
            with open(os.path.join(root, "prompt.json"), 'r') as file:
                lines = file.readlines()
            # check file existence for each line 
            for line in lines:
                json_data = json.loads(line)
                if os.path.exists(os.path.join(root, json_data["target"])) and os.path.exists(os.path.join(root, json_data["mask"])):
                    target_labels = label_map.get(os.path.splitext(json_data["target"].split("/")[-1])[0], None)
                    # read target
                    input_file = os.path.join(root, json_data["target"])
                    image_pil = Image.open(input_file).convert("RGB")
                    # read mask
                    input_file = os.path.join(root, json_data["mask"])
                    mask_pil = Image.open(input_file).convert("L")
                    # make detection
                    text_prompt = 'phone.' if target_labels["phone_class"] == 1 else 'cigarette.' if target_labels["cigarette_class"] == 1 else 'food. drink.' if target_labels["food_class"] == 1 else ''
                    obj_mask = torch.zeros((mask_pil.height, mask_pil.width), dtype=torch.bool)
                    if text_prompt != '':
                        masks, boxes, phrases, logits = model.predict(image_pil, text_prompt+" hand.")
                        idx_objects = [l in list(filter(lambda x: x != "" and x != ", ", text_prompt.split("."))) for i, l in enumerate(phrases) ]
                        hands = [boxes[i] for i, l in enumerate(phrases) if l in ["hand"]]
                        if any(idx_objects) and len(hands) != 0:
                            objects_identified[text_prompt] = objects_identified.get(text_prompt, 0)+1
                            segmented_objects = masks[idx_objects][check_overlap(torch.stack(hands), boxes[idx_objects])]
                            obj_mask = torch.cat([segmented_objects, obj_mask.unsqueeze(0)]).sum(0).bool()
                    # save detection
                    obj_mask_file = input_file.replace("mask", "obj_mask")
                    plt.imsave(obj_mask_file, obj_mask.cpu())
                    json_data["obj_mask"] = obj_mask_file.replace(root, "")
                    updated_lines.append(json_data)
        print(objects_identified)
        # overwrite the json
        with open(os.path.join(root, "prompt.json"), 'w') as outfile:
            for line in updated_lines:
                json.dump(line, outfile)
                outfile.write('\n')



def create_prompt_llava(args: argparse.Namespace):
    from llava_predictor import Predictor

    if os.path.exists(os.path.join(args.get("input_path", DATA_PATH), "target")):
        print("'target' folder will be used as input path.")
        args["input_path"] = os.path.join(args.get("input_path", DATA_PATH), "target")
    input_folder = args.get("input_path", DATA_PATH)+"/"
    root_out = args.get("output_path", DATA_PATH)+"/"
    mask_folder = os.path.join(root_out, "mask")
    target_folder = os.path.join(root_out, "target")
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    pred = Predictor()
    pred.setup()
    query = "follow this example of caption 'a man sitting in the driver's seat of a car' and provide a COMPACT and OBJECTIVE caption for the action currently performed by the driver if DISTRACTED, using OBJECTS or ATTENTIVE to the street"
    query = 'Provide as output ONLY THE BESTS label for the image choosing the list one of the following: "phone", "driver distracted", "driver drowsy", "driver attentive", "food or drink", "cigarette". "cigarette"=person interacting with a cigarette. "food or drink"=person interacting with foods or drinks. "phone"=person interacting with a phone. "driver attentive"=person focused watching forward with open eyes. "driver drowsy"= person with eyes closed or yawning (if drowsy not attentive or distracted). "driver distracted"=person not watching forward, or engaged in other activities, such as using a phone, eating, or smoking (if distracted not attentive or drowsy)'

    csv_file = args.get("csv", None)
    label_map = {}
    if csv_file is not None:
        with open(csv_file, "r") as file:
            csv_reader = csv.DictReader(file)
            for row in csv_reader:
                label_id = row["label_id"]
                phone_class = int(row["phone_binary"])
                cigarette_class = int(row["cigarette_binary"])
                food_class = int(row["food_binary"])
                label_map[label_id] = {"phone_class": phone_class, "cigarette_class": cigarette_class, "food_class": food_class}

    with open(os.path.join(root_out, 'prompt.json'), 'w') as outfile:
        print("Entering:", input_folder)
        for filename in tqdm(os.listdir(input_folder)):
            if not_image(filename): continue
            input_file = os.path.join(input_folder, filename)
            mask_file = os.path.join(mask_folder, filename)
            target_file = os.path.join(target_folder, filename)
            target_labels = label_map.get(os.path.splitext(filename)[0], None)

            if os.path.isfile(input_file) and os.path.exists(mask_file) and target_labels is not None:
                if input_file != target_file:
                    shutil.copyfile(input_file, target_file)

                item = "phone" if target_labels["phone_class"]==1 else "cigarette/cigar/vape" if target_labels["cigarette_class"] else "food/drink" if target_labels["food_class"] else "nothing"
                query = f"provide an OBJECTIVE caption for the action currently performed in the image by a person without hallucinating or making hypothesis out of what is visible, mention {item} in hand, make a sentence with present participle verbs and indeterminate article. Max 180 characters allowed"

                description = pred.predict(input_file, query, 0.7, 0.2, 512)
                description = (description[:-1] if description.endswith('.') else description).replace("\"", "").replace("'", "")
                line = {
                    "conditioning": mask_file.replace(root_out,""),
                    "mask": mask_file.replace(root_out,""),
                    "target": target_file.replace(root_out,""),
                    "prompt": description+f", {item} in hand",
                    **target_labels
                }
                json.dump(line, outfile)
                outfile.write('\n')


def create_prompt_blip(args: argparse.Namespace):
    from transformers import pipeline

    root = args.get("output_path", DATA_PATH)

    conditioning_dir = os.path.join(root, "poses")
    mask_dir = os.path.join(root, "mask")
    target_dir = os.path.join(root, "target")
    pred = pipeline("image-to-text", model="Salesforce/blip-image-captioning-large", device=0)

    with open(os.path.join(root, 'prompt_blip.json'), 'w') as outfile:
        for folder_name in os.listdir(target_dir):
            mask_folder = os.path.join(mask_dir, folder_name)
            target_folder = os.path.join(target_dir, folder_name)

            if os.path.isdir(target_folder):
                print("Entering:", target_folder)
                for filename in tqdm(os.listdir(target_folder)):
                    if not_image(filename): continue
                    mask_file = os.path.join(mask_folder, filename)
                    target_file = os.path.join(target_folder, filename)

                    if os.path.isfile(target_file):
                        label = pred(target_file)[0]['generated_text']
                        label = (label[:-1] if label.endswith('.') else label)+", extremely detailed, photorealistic."
                        label = label.replace("\"", "").replace("'", "")
                        line = {"conditioning": mask_file.replace(root,""),
                                  "mask": mask_file.replace(root,""),
                                  "target": target_file.replace(root,""),
                                  "prompt": label}
                        json.dump(line, outfile)
                        outfile.write('\n')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--action", required=True, choices=['mask', 'pose', 'prompt', 'clean', 'obj_mask'], help="Choose which action to run on the data.")
    ap.add_argument("--input_path", help="Path with images to process")
    ap.add_argument("--output_path", help="Path where to save the processed output")
    ap.add_argument("--csv", help="Path where to find associations between images and labels")
    args = vars(ap.parse_args())

    if args.get("action") == "mask":
        run_masking(args)
    if args.get("action") == "pose":
        run_pose_estimation(args)
    if args.get("action") == "prompt":
        create_prompt_llava(args)
    if args.get("action") == "clean":
        clean_preprocessed_data(args)
    if args.get("action") == "obj_mask":
        run_object_detection(args)
