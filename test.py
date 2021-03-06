from utils import detect_hand
import cv2
import numpy as np
import os

toad = cv2.imread('./data/toads/' + np.random.choice(os.listdir('./data/toads/')), cv2.IMREAD_UNCHANGED)
image = cv2.imread('./data/detector_train_data/test/2_pic00007.jpg')

result, len = detect_hand.predict_and_draw(image, toad)

cv2.imwrite("result.jpg", result)