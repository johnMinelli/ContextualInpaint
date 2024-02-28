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
from tqdm import tqdm
from sys import platform

def not_image(filename: str):
    return not filename.lower().endswith(".jpeg") and not filename.lower().endswith(".jpg") and not filename.lower().endswith(".png")

# from densepose import add_densepose_config
# from detectron2.projects.DensePose.apply_net import ShowAction

DATA_PATH = "../data/verizon_formatted2/"

def clean_preprocessed_data(args):
    def remove(files):
        for f in files:
            os.remove(f)

    dil_kernel = np.ones((7, 7), np.uint8)  # Adjust the kernel size according to your requirements
    for root, dirs, files in os.walk(os.path.join(args.get("input_path", DATA_PATH), "source")):
        if len(files)==0: continue
        print("Entering folder:", root)
        for filename in tqdm(files):
            if not_image(filename): continue
            rgb_path = os.path.join(root, filename)
            mask_path = os.path.join(root.replace("source", "mask"), filename)
            pose_path = os.path.join(root.replace("source", "poses"), filename)
            try:
                rgb_im = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
                mask_im = cv2.cvtColor(cv2.imread(mask_path), cv2.COLOR_BGR2RGB)
                pose_im = cv2.cvtColor(cv2.imread(pose_path), cv2.COLOR_BGR2RGB)
                shape = rgb_im.shape
            except Exception as e:
                print("ERR", os.path.join(root, filename), str(e))
                continue
            if np.any(mask_im==255):
                # right crop
                rgb_im = rgb_im[:,-shape[0]:,:]
                mask_im = mask_im[:,-shape[0]:,:]
                pose_im = pose_im[:,-shape[0]:,:]
                if np.sum(mask_im[:,0,0]>50)/shape[0] < 0.5 and (np.sum(mask_im[:,:,0]>50)/(shape[0]*shape[1]))>0.15:
                    # resize
                    rgb_im = cv2.resize(rgb_im, (512, 512))
                    mask_im = cv2.resize(mask_im, (512, 512))
                    pose_im = cv2.resize(pose_im, (512, 512))
                    # # dilate mask and apply
                    # bool_mask = cv2.dilate((mask_im>50).astype(np.uint8) * 255, dil_kernel, iterations=2) > 0
                    # rgb_im[bool_mask] = 0
                    # pose_im[~bool_mask] = 0
                    # mask_im = np.zeros_like(rgb_im)
                    # mask_im[bool_mask] = 255
                    # save
                    cv2.imwrite(rgb_path, rgb_im)
                    cv2.imwrite(mask_path, mask_im)
                    cv2.imwrite(pose_path, pose_im)
                else:
                    remove([rgb_path, mask_path, pose_path])
            else:
                remove([rgb_path, mask_path, pose_path])


def run_masking(args: argparse.Namespace):
    cfg = get_cfg()
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.4  # set threshold for this model
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    predictor = DefaultPredictor(cfg)
    dil_kernel = np.ones((7, 7), np.uint8)  # Adjust the kernel size according to your requirements

    # Create output folder if not exist
    source_folder = os.path.join(args.get("output_path", DATA_PATH), "source")
    mask_folder = os.path.join(args.get("output_path", DATA_PATH), "mask")
    if not os.path.exists(source_folder):
        os.makedirs(source_folder)
    if not os.path.exists(mask_folder):
        os.makedirs(mask_folder)

    # Create masked source files
    if os.path.exists(os.path.join(args.get("input_path", DATA_PATH), "target")):
        print("'target' folder will be used as input path.")
        args.input_path = os.path.join(args.get("input_path", DATA_PATH), "target")
    for root, dirs, files in os.walk(args.get("input_path", DATA_PATH)):
        if len(files)==0: continue
        print("Entering folder:", root)
        for filename in tqdm(files):
            if not_image(filename): continue      

            try:
                rgb_im = cv2.imread(os.path.join(root, filename))
                rgb_im = cv2.cvtColor(rgb_im, cv2.COLOR_BGR2RGB)
            except Exception as e:
                print("ERR", os.path.join(root, filename), str(e))
                continue
            outputs = predictor(rgb_im)
            # v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
            # out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            # plt.imshow(pred_masks)
            # plt.show()
            person_filter = outputs["instances"].pred_classes==0
            pred_masks = outputs["instances"].pred_masks[person_filter].cpu()
            pred_mask_bool = pred_masks.sum(0).bool()
            pred_mask_bool = cv2.dilate(pred_mask_bool.numpy().astype(np.uint8) * 255, dil_kernel, iterations=2) > 0

            rgb_im[pred_mask_bool] = [0, 0, 0]
            mask_im = np.zeros_like(rgb_im)
            mask_im[pred_mask_bool] = [255, 255, 255]

            # Save
            plt.imsave(os.path.join(source_folder, filename), rgb_im)
            plt.imsave(os.path.join(mask_folder, filename), mask_im)


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
        args.input_path = os.path.join(args.get("input_path", DATA_PATH), "target")
    for root, dirs, files in os.walk(args.get("input_path", DATA_PATH)):
        if len(files)==0: continue
        print("Entering:", root)
        for filename in tqdm(files):
            if not_image(filename): continue
            
            try:
                im = cv2.imread(os.path.join(root, filename))
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


    for root, dirs, files in os.walk(os.path.join(args.get("input_path", DATA_PATH),"target")):
        if len(files)==0: continue
        print("Entering:", root)
        for filename in tqdm(files):
            if not_image(filename): continue
            im = cv2.imread(os.path.join(root,filename))
            datum = op.Datum()
            datum.cvInputData = im
            opWrapper.emplaceAndPop(op.VectorDatum([datum]))
        
            source_folder = root.replace("target", "openposes")
            if not os.path.exists(source_folder):
                os.makedirs(source_folder)
            # Save
            plt.imsave(os.path.join(source_folder, filename), datum.cvOutputData)

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
    for root, dirs, files in os.walk(os.path.join(args.get("input_path", DATA_PATH), "target")):
        if len(files)==0: continue
        print("Entering:", root)
        for filename in tqdm(files):
            if not_image(filename): continue
            im = cv2.imread(os.path.join(root,filename))
            
            
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

            ShowAction.execute_on_outputs(context, {"file_name": os.path.join(source_folder, filename), "image": im}, outputs)
            ShowAction.postexecute(context)

    torch.save(poses, args.get("output_path", DATA_PATH), "poses.txt")

def create_prompt_llava(args: argparse.Namespace):
    from llava_predictor import Predictor
    root = args.get("input_path", DATA_PATH)+"/"

    # TODO make it more flexible
    assert os.path.join(root, "poses") and os.path.join(root, "mask") and os.path.join(root, "target"), "The input path provided should present folders 'target', 'source' and 'poses'"

    conditioning_folder = conditioning_dir = os.path.join(root, "poses")
    mask_folder = mask_dir = os.path.join(root, "mask")
    target_folder = target_dir = os.path.join(root, "target")
    pred = Predictor()
    pred.setup()
    query = "follow this example of caption 'a man sitting in the driver's seat of a car' and provide a COMPACT and OBJECTIVE caption for the action currently performed by the driver if DISTRACTED, using OBJECTS or ATTENTIVE to the street"
    query = 'Provide as output ONLY THE BESTS label for the image choosing the list one of the following: "phone", "driver distracted", "driver drowsy", "driver attentive", "food or drink", "cigarette". "cigarette"=person interacting with a cigarette. "food or drink"=person interacting with foods or drinks. "phone"=person interacting with a phone. "driver attentive"=person focused watching forward with open eyes. "driver drowsy"= person with eyes closed or yawning (if drowsy not attentive or distracted). "driver distracted"=person not watching forward, or engaged in other activities, such as using a phone, eating, or smoking (if distracted not attentive or drowsy)'
    query = "provide a COMPACT and OBJECTIVE caption for the action currently performed in the image by a person without hallucinating or making hypothesis out is visible, use noun phrase with present participle verbs and indeterminate article. Maximum 180 characters allowed"

    with open(os.path.join(root, 'prompt.json'), 'w') as outfile:
        # for folder_name in os.listdir(target_dir):
            # conditioning_folder = os.path.join(conditioning_dir, folder_name)
            # mask_folder = os.path.join(mask_dir, folder_name)
            # target_folder = os.path.join(target_dir, folder_name)
            # 
            # if os.path.isdir(conditioning_folder) and os.path.isdir(target_folder):
                print("Entering:", target_folder)
                for filename in tqdm(os.listdir(target_folder)):
                    if not_image(filename): continue
                    conditioning_file = os.path.join(conditioning_folder, filename)
                    mask_file = os.path.join(mask_folder, filename)
                    target_file = os.path.join(target_folder, filename)
        
                    if os.path.isfile(conditioning_file) and os.path.isfile(target_file):
                        label = pred.predict(target_file, query, 0.7, 0.2, 512)
                        label = (label[:-1] if label.endswith('.') else label).replace("\"", "").replace("'", "")
                        line = {
                            "conditioning": conditioning_file.replace(root,""),
                            "mask": mask_file.replace(root,""),
                            "target": target_file.replace(root,""),
                            "prompt": label
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
            conditioning_folder = os.path.join(conditioning_dir, folder_name)
            mask_folder = os.path.join(mask_dir, folder_name)
            target_folder = os.path.join(target_dir, folder_name)

            if os.path.isdir(conditioning_folder) and os.path.isdir(target_folder):
                print("Entering:", target_folder)
                for filename in tqdm(os.listdir(target_folder)):
                    if not_image(filename): continue
                    conditioning_file = os.path.join(conditioning_folder, filename)
                    mask_file = os.path.join(mask_folder, filename)
                    target_file = os.path.join(target_folder, filename)
        
                    if os.path.isfile(conditioning_file) and os.path.isfile(target_file):
                        label = pred(target_file)[0]['generated_text']
                        label = (label[:-1] if label.endswith('.') else label)+", extremely detailed, photorealistic."
                        label = label.replace("\"", "").replace("'", "")
                        line = {"conditioning": conditioning_file.replace(root,""),
                                  "mask": mask_file.replace(root,""),
                                  "target": target_file.replace(root,""),
                                  "prompt": label}
                        json.dump(line, outfile)
                        outfile.write('\n')

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--action", required=True, choices=['mask', 'pose', 'prompt', 'clean'], help="Choose which action to run on the data.")
    ap.add_argument("--input_path", help="Path with images to process")
    ap.add_argument("--output_path", help="Path where to save the processed output")
    args = vars(ap.parse_args())

    if args.get("action") == "mask":
        run_masking(args)
    if args.get("action") == "pose":
        run_pose_estimation(args)
    if args.get("action") == "prompt":
        create_prompt_llava(args)
    if args.get("action") == "clean":
        clean_preprocessed_data(args)
