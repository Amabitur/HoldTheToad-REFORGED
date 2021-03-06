from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
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
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2
cfg.MODEL.DEVICE = "cpu"

predictor = DefaultPredictor(cfg)


def get_hand_prediction(image):
    outputs = predictor(image)
    print(outputs)
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
        toad = correct_toad_color(image[ymin:ymax, xmin:xmax], toad)
        cv2.imwrite("hand.jpg", image[ymin:ymax, xmin:xmax])
        perc0 = 0.75*min(xmax - xmin, ymax - ymin) / max(toad.shape)
        w0 = int(toad.shape[1] * perc0)
        h0 = int(toad.shape[0] * perc0)
        toad = cv2.resize(toad, (w0, h0))
        pilimg = Image.fromarray(image)
        piltoad = Image.fromarray(toad)
        pilimg.paste(piltoad, (xmin, ymin + (ymax - ymin) // 4), piltoad)
        image = np.array(pilimg)
    return image


def correct_toad_color(hand, toad):
    pre_hand = cv2.cvtColor(hand.copy(), cv2.COLOR_BGR2HSV)
    pre_toad = cv2.cvtColor(toad[:, :, 0:3].copy(), cv2.COLOR_BGR2HSV)
    satur_coef = pre_toad[:, :, 1].mean()/pre_hand[:, :, 1].mean()

    pre_toad[:, :, 1] = pre_toad[:, :, 1]/satur_coef
    new_toad = cv2.cvtColor(np.uint8(pre_toad), cv2.COLOR_HSV2BGR)

    new_toad_alpha = toad.copy()
    new_toad_alpha[:, :, 0] = new_toad[:, :, 0]
    new_toad_alpha[:, :, 1] = new_toad[:, :, 1]
    new_toad_alpha[:, :, 2] = new_toad[:, :, 2]

    return new_toad_alpha

    return new_toad_alpha


def predict_and_draw(image, toad):
    boxes = get_hand_prediction(image)[:1]
    print(boxes)
    print(len(boxes))
    # img = draw_boxes(image, boxes)
    img = draw_toad(image, boxes, toad)
    return img, len(boxes)
