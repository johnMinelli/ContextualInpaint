import argparse
import json
import os
import warnings
from dataclasses import dataclass


import cv2
import matplotlib.pyplot as plt

import detectron2
import numpy as np
import torch
import sys
from PIL import Image
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from jinja2.nodes import Output
from pyasn1_modules.rfc2315 import Data
from tqdm import tqdm
from sys import platform
# from densepose import add_densepose_config

# from detectron2.projects.DensePose.apply_net import ShowAction
from preprocess.prompt_llava import Predictor

DATA_PATH = "../data/verizon_formatted2/"

def run_masking(args: argparse.Namespace):
    cfg = get_cfg()
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.4  # set threshold for this model
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    predictor = DefaultPredictor(cfg)

    # Create masked source files
    for root, dirs, files in os.walk(os.path.join(args.get("input_path", DATA_PATH), "target")):
        if len(files)==0: continue
        print("Entering folder:", root)
        for file in tqdm(files):
            if not file.endswith(".jpg") and not file.endswith(".png"): continue
            rgb_im = cv2.imread(os.path.join(root, file))
            rgb_im = cv2.cvtColor(rgb_im, cv2.COLOR_BGR2RGB)
            outputs = predictor(rgb_im)
            # v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
            # out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            # plt.imshow(pred_masks)
            # plt.show()
            person_filter = outputs["instances"].pred_classes==0
            pred_masks = outputs["instances"].pred_masks[person_filter].cpu()
            pred_mask_bool = pred_masks.sum(0).bool()

            # # Parse the XML file (Stanford dataset tweak)
            # tree = ET.parse(os.path.join(root,file.replace(".jpg", ".xml")))
            # xmlroot = tree.getroot()
            # # Extract bounding box coordinates from the XML
            # xmin = int(int(xmlroot.find('object/bndbox/xmin').text) * (512 / int(xmlroot.find('size/width').text)))
            # ymin = int(int(xmlroot.find('object/bndbox/ymin').text) * (512 / int(xmlroot.find('size/height').text)))
            # xmax = int(int(xmlroot.find('object/bndbox/xmax').text) * (512 / int(xmlroot.find('size/width').text)))
            # ymax = int(int(xmlroot.find('object/bndbox/ymax').text) * (512 / int(xmlroot.find('size/height').text)))
            # 
            # # Apply boolean mask to filter predictions
            # inv = np.zeros_like(pred_mask_bool, dtype=bool)
            # inv[ymin:ymax, xmin:xmax] = True
            # pred_mask_bool = (pred_mask_bool * inv).bool()

            rgb_im[pred_mask_bool] = [0, 0, 0]
            mask_im = np.zeros_like(rgb_im)
            mask_im[pred_mask_bool] = [255, 255, 255]

            # Create a folder with the activity name if it doesn't exist
            source_folder = os.path.join(args.get("output_path", DATA_PATH), "source")
            mask_folder = os.path.join(args.get("output_path", DATA_PATH), "mask")
            if not os.path.exists(source_folder):
                os.makedirs(source_folder)
            if not os.path.exists(mask_folder):
                os.makedirs(mask_folder)
            # Save
            plt.imsave(os.path.join(source_folder, file), rgb_im)
            plt.imsave(os.path.join(mask_folder, file), mask_im)

def run_pose_estimation(args: argparse.Namespace):
    cfg = get_cfg()  # get a fresh new config
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.65  # set threshold for this model
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml")
    predictor = DefaultPredictor(cfg)

    poses = {}
    for root, dirs, files in os.walk(os.path.join(args.get("input_path", DATA_PATH), "target")):
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

            v = Visualizer(np.zeros_like(im), MetadataCatalog.get(cfg.DATASETS.TRAIN[0]))
            out = v.draw_instance_predictions(outputs).get_image()[:, :, ::-1]
            # plt.imshow(out)
            # plt.show()            
            poses_folder = os.path.join(args.get("output_path", DATA_PATH), "poses")
            if not os.path.exists(poses_folder):
                os.makedirs(poses_folder)
            # Save
            plt.imsave(os.path.join(poses_folder, file), out)

    torch.save(poses, DATA_PATH+"poses.txt")

def run_openpose_estimation(args: argparse.Namespace):
    # Openpose Import (Windows/Ubuntu/OSX)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    try:
        # Windows Import
        if platform == "win32":
            # Change these variables to point to the correct folder (Release/x64 etc.)
            sys.path.append(dir_path + '/../../python/openpose/Release');
            os.environ['PATH'] = os.environ['PATH'] + ';' + dir_path + '/../../x64/Release;' + dir_path + '/../../bin;'
            import pyopenpose as op
        else:
            # Change these variables to point to the correct folder (Release/x64 etc.)
            sys.path.append('../../python')
            # If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
            # sys.path.append('/usr/local/python')
            from openpose import pyopenpose as op
    except ImportError as e:
        print(
            'Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
        raise e
    
    
    
    params = dict()
    params["model_folder"] = "../../../models/"
    params["face"] = True
    params["hand"] = True
    
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()


    for root, dirs, files in os.walk(DATA_PATH+"target"):
        if len(files)==0: continue
        print("Entering:", root)
        for file in tqdm(files):
            if not file.endswith(".jpg"): continue
            im = cv2.imread(os.path.join(root,file))
            datum = op.Datum()
            datum.cvInputData = im
            opWrapper.emplaceAndPop(op.VectorDatum([datum]))
        
            source_folder = root.replace("target", "openposes")
            if not os.path.exists(source_folder):
                os.makedirs(source_folder)
            # Save
            plt.imsave(os.path.join(source_folder, file), datum.cvOutputData)

    # torch.save(poses, DATA_PATH+"poses.txt")

def run_densepose_estimation(args: argparse.Namespace):
    @dataclass
    class DensePoseSettings:
        visualizations: str
        score: str
        texture_atlas: str
        texture_atlases_map: str
        image_file: str

    args = DensePoseSettings(visualizations="dp_u")
    
    cfg = get_cfg()  # get a fresh new config
    add_densepose_config(cfg)

    cfg.merge_from_file(model_zoo.get_config_file("DensePose/densepose_rcnn_R_101_FPN_s1x.yaml"))
    cfg.MODEL.WEIGHTS = "../detectron2/models/densepose_rcnn_R_101_FPN_s1x.pkl"
    predictor = DefaultPredictor(cfg)
    context = ShowAction.create_context(args, cfg)
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
      
            source_folder = root.replace("target", "denseposes")
            if not os.path.exists(source_folder):
                os.makedirs(source_folder)

            ShowAction.execute_on_outputs(context, {"file_name": os.path.join(source_folder, file), "image": im}, outputs)
            ShowAction.postexecute(context)

    torch.save(poses, DATA_PATH+"poses.txt")

def create_prompt_json(args: argparse.Namespace):

    conditioning_dir = os.path.join(DATA_PATH, 'poses')
    mask_dir = os.path.join(DATA_PATH, 'mask')
    target_dir = os.path.join(DATA_PATH, 'target')
    pred = Predictor()
    pred.setup()
    query = "follow this example of caption 'a man sitting in the driver's seat of a car' and provide a COMPACT and OBJECTIVE caption for the action currently performed by the driver if DISTRACTED, using OBJECTS or ATTENTIVE to the street"
    query = "provide a COMPACT and OBJECTIVE caption for the action currently performed in the action by a person without hallucinating or making hyothesis out of what you see, use noun phrase with present participle verbs and indeterminative article"
    query = 'Provide as output ONLY THE BESTS label for the image choosing the list one of the following: "phone", "driver distracted", "driver drowsy", "driver attentive", "food or drink", "cigarette". "cigarette"=person interacting with a cigarette. "food or drink"=person interacting with foods or drinks. "phone"=person interacting with a phone. "driver attentive"=person focused watching forward with open eyes. "driver drowsy"= person with eyes closed or yawning (if drowsy not attentive or distracted). "driver distracted"=person not watching forward, or engaged in other activities, such as using a phone, eating, or smoking (if distracted not attentive or drowsy)'

    with open(os.path.join(DATA_PATH, 'prompt.json'), 'w') as outfile:
        for folder_name in os.listdir(target_dir):
            conditioning_folder = os.path.join(conditioning_dir, folder_name)
            mask_folder = os.path.join(mask_dir, folder_name)
            target_folder = os.path.join(target_dir, folder_name)

            if os.path.isdir(conditioning_folder) and os.path.isdir(target_folder):
            # conditioning_folder = conditioning_dir
            # mask_folder = mask_dir
            # target_folder = target_dir
                print("Entering:", target_folder)
                for filename in tqdm(os.listdir(target_folder)):
                    if not filename.endswith(".jpg"): continue
                    conditioning_file = os.path.join(conditioning_folder, filename)
                    mask_file = os.path.join(mask_folder, filename)
                    target_file = os.path.join(target_folder, filename)
        
                    if os.path.isfile(conditioning_file) and os.path.isfile(target_file):
                        label = pred.predict(target_file, query, 0.7, 0.2, 512)
                        label = label[:-1] if label.endswith('.') else label
                        label = label.replace("\"", "").replace("'", "")
                        prompt = {"control": conditioning_file.replace(DATA_PATH,""),
                                  "mask": mask_file.replace(DATA_PATH,""),
                                  "target": target_file.replace(DATA_PATH,""),
                                  "prompt": label}
                        json.dump(prompt, outfile)
                        outfile.write('\n')

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--action", required=True, choices=['mask', 'pose', 'densepose', 'prompt'], help="Choose which action to run on the data.")
    ap.add_argument("--input_path", help="Path with images to process")
    ap.add_argument("--output_path", help="Path where to save the processed output")
    args = vars(ap.parse_args())

    if args.get("action") == "mask":
        run_masking(args)
    elif args.get("action") == "pose":
        run_pose_estimation(args)
    elif args.get("action") == "prompt":
        create_prompt_json(args)
    elif args.get("action") == "densepose":
        run_densepose_estimation(args)
