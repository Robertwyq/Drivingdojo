from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
import cv2
import os
import numpy as np
import tqdm
from argparse import ArgumentParser

# import pdb
# pdb.set_trace()

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.1  # set threshold for this model
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
predictor = DefaultPredictor(cfg)

parser = ArgumentParser("maskrcnn")
parser.add_argument("--image_folder", default="outputs/images", type=str)
parser.add_argument("--output_folder", default="outputs/masks", type=str)
args = parser.parse_args()

meta_folder = args.image_folder
meta_output_folder = args.output_folder
if not os.path.exists(meta_output_folder):
    os.makedirs(meta_output_folder)
for scene in tqdm.tqdm(sorted(os.listdir(meta_folder))):

    if '.pkl' in scene or 'trajs' in scene:
        continue

    input_folder = os.path.join(meta_folder,scene)
    output_folder = os.path.join(meta_output_folder,scene)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # COCO dataset class IDs for person, car, and bicycle
    target_classes = [0, 1, 2, 3 , 4, 5, 6 , 7 , 8]  # person, bicycle, car

    for image_name in sorted(os.listdir(input_folder)):
        image_path = os.path.join(input_folder, image_name)
        img = cv2.imread(image_path)
        outputs = predictor(img)
        
        # 获取对应类别的mask
        instances = outputs["instances"]
        masks = instances.pred_masks
        classes = instances.pred_classes

        # 创建一个空的mask图像
        combined_mask = np.zeros(img.shape[:2], dtype=np.uint8)
        
        for i, mask in enumerate(masks):
            if classes[i] in target_classes:
                combined_mask = np.maximum(combined_mask, mask.cpu().numpy().astype(np.uint8) * 255)

        # import pdb
        # pdb.set_trace()

        combined_mask = 255-combined_mask

        # 保存mask图像
        output_path = os.path.join(output_folder, f"{image_name}.png")
        cv2.imwrite(output_path, combined_mask)
        # import pdb
        # pdb.set_trace()