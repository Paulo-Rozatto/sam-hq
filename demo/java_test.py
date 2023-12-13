import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from segment_anything import sam_model_registry, SamPredictor
import os
import sys

# sam_checkpoint = "/home/paulo/Desktop/tcc/sam-hq/pretrained_checkpoint/sam_hq_vit_tiny.pth"
# model_type = "vit_tiny"
# sam_checkpoint = "/home/paulo/Desktop/tcc/sam-hq/pretrained_checkpoint/sam_hq_vit_b.pth"
sam_checkpoint = "/home/paulo/Desktop/tcc/sam-hq/pretrained_checkpoint/test_10.pth"
model_type = "vit_b"
device = "cpu"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
sam.eval()
predictor = SamPredictor(sam)

imgePath = sys.argv[1]
if not os.path.exists(imgePath):
    print("Image not found")
    sys.exit(1)
image = cv2.imread(imgePath)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
predictor.set_image(image)
hq_token_only = True

# point prompt
points = sys.argv[2]
if type(points) is not str:
    print("Invalid points")
    sys.exit(1)
points = [int(x) for x in points.split(",")]

# input_point = np.array([[2125,699],[2709,774], [2314,2395]]) #,[2314,2395], [2393, 1065]])
input_point = None #np.array(points)
input_label = None # np.ones(input_point.shape[0])
input_box = np.array([points])
masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    box = input_box,
    multimask_output=False,
    hq_token_only=hq_token_only,
)

# leaf_path = "/home/paulo/Desktop/tcc/sam-hq/demo/leaf_prediction"
# mask_path = os.path.join(leaf_path, os.path.basename(imgePath))
# mask = masks[0]
# h, w = mask.shape[-2:]
# mask_image = mask.reshape(h, w) * 255
# cv2.imwrite(mask_path, mask_image)


# find contours from mask, draw and show image
contours, _ = cv2.findContours(masks[0].astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
biggest_contour = max(contours, key=cv2.contourArea)
poly = cv2.approxPolyDP(biggest_contour, 0.00054 * cv2.arcLength(biggest_contour, True), True)
poly = poly.reshape(poly.shape[0], poly.shape[2]).tolist()
print(poly)
