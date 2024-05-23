import numpy as np
import cv2
from PIL import Image
import json
import os
import matplotlib.pyplot as plt
import shutil
import random

f = open('./train.json')
data = json.load(f)
f.close()

first_ori_image = './removed_train/25.png'
second_ori_image = './removed_train/132.png'
third_ori_image = './removed_train/243.png'
fourth_ori_image = './removed_train/357.png'
# 첫번째 배경: 0 ~ 61, 기준 25 / 29 사용
# 두번째 배경: 124~179, 기준 132 / 129 사용
# 세번째 배경: 237~283, 기준 243 / 270 사용
# 네번째 배경: 332~393, 기준 357 / 376 사용
image_dir = './asdf/' # './yolo_dataset/images/train/'
image_files = [file for file in os.listdir(image_dir) if file.endswith('.png')]

def create_segmentation_mask(image, segmentation_info):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for obj in segmentation_info:
        polygon = np.array(obj['segmentation'], dtype=np.int32)
        cv2.fillPoly(mask, [polygon], 255)
    return mask

def remove_person_and_make_black(image, segmentation_info):
    # Create the segmentation mask
    mask = create_segmentation_mask(image, segmentation_info)
    
    # Set the masked area to black
    image[mask == 255] = 0
    
    return image

def inpaint_image(image, mask):
    inpainted_image = cv2.inpaint(image, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    return inpainted_image

def fill_mask_with_other_images(base_image, mask, image_dir, exclude_image):
    for filename in os.listdir(image_dir):
        if filename != exclude_image:
            other_image_path = os.path.join(image_dir, filename)
            other_image = cv2.imread(other_image_path)
            # Fill the mask area with the other image if the pixel is not black
            base_image[(mask == 255) & (other_image != 0).all(axis=2)] = other_image[(mask == 255) & (other_image != 0).all(axis=2)]
    return base_image

base_image = cv2.imread(fourth_ori_image)

for image_file in image_files:
    image_path = os.path.join(image_dir, image_file)
    image = cv2.imread(image_path)
    image_id = 357
    segmentation_info = []
    for ann in data['annotations']:
        if ann['image_id'] == image_id:
            person_id = ann['category_id']
            segmentation = ann['segmentation']
            segmentation_info.append({'category_id': person_id, 'segmentation': segmentation})
    mask = create_segmentation_mask(image, segmentation_info)
    
    result_image = fill_mask_with_other_images(base_image, mask, image_dir, '357.png')
    cv2.imwrite('test.png', result_image)
    exit()
    