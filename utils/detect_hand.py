from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
import os
import cv2
import numpy as np
from PIL import Image

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 300
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.MODEL.WEIGHTS = "data/models/model_final.pth"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
cfg.MODEL.DEVICE = "cpu"

predictor = DefaultPredictor(cfg)


def get_hand_prediction(image):
    outputs = predictor(image)
    boxes = outputs['instances'].pred_boxes
    return boxes.tensor.numpy().astype(int)


def draw_boxes(image, boxes):
    for box in boxes:
        xmin, ymin, xmax, ymax = box
        image = cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
    return image

def draw_toad(image, boxes, toad):
    for box in boxes:
        xmin, ymin, xmax, ymax = box
        perc0 = min(xmax-xmin, ymax-ymin) / max(toad.shape)
        w0 = int(toad.shape[1] * perc0)
        h0 = int(toad.shape[0] * perc0)
        toad = cv2.resize(toad, (w0, h0))
        pilimg = Image.fromarray(image)
        piltoad = Image.fromarray(toad)
        pilimg.paste(piltoad, (xmin, (ymin+ymax)//4), piltoad)
        image = np.array(pilimg)
    return image


image = cv2.imread("data/hands/Human-Hands-Front-Back.jpg")
toad = cv2.imread("data/toads/toad1.png", cv2.IMREAD_UNCHANGED)

boxes = get_hand_prediction(image)
img = draw_boxes(image, boxes)
img = draw_toad(img, boxes, toad)
cv2.imwrite("toad_img.jpg", img)