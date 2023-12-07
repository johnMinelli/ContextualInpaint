# !python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
import argparse
import json
import os
import warnings
from pdb import run

import cv2
import matplotlib.pyplot as plt

# import some common detectron2 utilities
import detectron2
import numpy as np
import torch
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from jinja2.nodes import Output
from pyasn1_modules.rfc2315 import Data
from tqdm import tqdm

# from preprocess.prompt_llava import Predictor

DATA_PATH = "../data/verizon_formatted/"

def run_masking():
    cfg = get_cfg()
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    predictor = DefaultPredictor(cfg)

    # Create masked source files
    for root, dirs, files in os.walk(DATA_PATH+"target"):
        if len(files)==0: continue
        print("Entering:", root)
        for file in tqdm(files):
            if not file.endswith(".jpg"): continue
            rgb_im = cv2.imread(os.path.join(root,file))
            outputs = predictor(rgb_im)
                # v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
            # out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            # plt.imshow(pred_masks)
            # plt.show()
            person_filter = outputs["instances"].pred_classes==0
            pred_masks = outputs["instances"].pred_masks[person_filter].cpu()
            pred_mask_bool = pred_masks.sum(0).bool()

            # blur_im = cv2.GaussianBlur(im, (67, 67), 0)
            # rgb_im[pred_mask_bool] = blur_im[pred_mask_bool]
            # rgb_im[pred_mask_bool] = [235, 255, 7]  # EBFF07
            rgb_im[pred_mask_bool] = [0, 0, 0]
            mask_im = np.zeros_like(rgb_im)
            mask_im[pred_mask_bool] = [255, 255, 255]

            # Create a folder with the activity name if it doesn't exist
            source_folder = root.replace("target", "source")
            mask_folder = root.replace("target", "mask")
            if not os.path.exists(source_folder):
                os.makedirs(source_folder)
            if not os.path.exists(mask_folder):
                os.makedirs(mask_folder)
            # Save
            plt.imsave(os.path.join(source_folder, file), rgb_im)
            plt.imsave(os.path.join(mask_folder, file), mask_im)

def run_pose_estimation():
    cfg = get_cfg()  # get a fresh new config
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set threshold for this model
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")
    predictor = DefaultPredictor(cfg)

    poses = {}
    for root, dirs, files in os.walk(DATA_PATH+"target"):
        if len(files)==0: continue
        print("Entering:", root)
        for file in tqdm(files):
            if not file.endswith(".jpg"): continue
            im = cv2.imread(os.path.join(root,file))
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

            # poses[file] = outputs

            v = Visualizer(np.zeros_like(im), MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
            out = v.draw_instance_predictions(outputs).get_image()[:, :, ::-1]
            # plt.imshow(out)
            # plt.show()
            source_folder = root.replace("target", "poses")
            if not os.path.exists(source_folder):
                os.makedirs(source_folder)
            # Save
            plt.imsave(os.path.join(source_folder, file), out)

    torch.save(poses, DATA_PATH+"poses.txt")

def create_prompt_json():

    conditioning_dir = os.path.join(DATA_PATH, 'poses')
    mask_dir = os.path.join(DATA_PATH, 'mask')
    target_dir = os.path.join(DATA_PATH, 'target')
    pred = Predictor()
    pred.setup()
    query = "follow this example of caption 'a man sitting in the driver's seat of a car' and provide a COMPACT and OBJECTIVE caption for the action currently performed by the driver if DISTRACTED, using OBJECTS or ATTENTIVE to the street"

    with open(os.path.join(DATA_PATH, 'prompt.json'), 'w') as outfile:
        for folder_name in os.listdir(target_dir):
            conditioning_folder = os.path.join(conditioning_dir, folder_name)
            mask_folder = os.path.join(mask_dir, folder_name)
            target_folder = os.path.join(target_dir, folder_name)

            if os.path.isdir(conditioning_folder) and os.path.isdir(target_folder):
                print("Entering:", target_folder)
                for filename in tqdm(os.listdir(target_folder)):
                    conditioning_file = os.path.join(conditioning_folder, filename)
                    mask_file = os.path.join(mask_folder, filename)
                    target_file = os.path.join(target_folder, filename)

                    if os.path.isfile(conditioning_file) and os.path.isfile(target_file):
                        label = pred.predict(target_file, query, 0.7, 0.2, 512)
                        label = (label[:-1] if label.endswith('.') else label)+", best quality, extremely detailed"
                        prompt = {"control": conditioning_file.replace(DATA_PATH,""),
                                  "mask": mask_file.replace(DATA_PATH,""),
                                  "target": target_file.replace(DATA_PATH,""),
                                  "prompt": label}
                        json.dump(prompt, outfile)
                        outfile.write('\n')

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--action", required=True, choices=['mask', 'pose', 'prompt'], help="Choose which action run on the data.")
    args = vars(ap.parse_args())

    if args.get("action") == "mask":
        run_masking()
    elif args.get("action") == "pose":
        run_pose_estimation()
    elif args.get("action") == "prompt":
        create_prompt_json()
