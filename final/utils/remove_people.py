import numpy as np
import cv2
from PIL import Image
import json
import os
import matplotlib.pyplot as plt
import shutil
import random

def create_segmentation_mask(image, segmentation_info):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for obj in segmentation_info:
        polygon = np.array(obj['segmentation'], dtype=np.int32)
        cv2.fillPoly(mask, [polygon], 255)
    return mask

def remove_person_and_make_transparent(image, segmentation_info):
    # Convert image to RGBA format
    rgba_image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    
    # Create the segmentation mask
    mask = create_segmentation_mask(image, segmentation_info)
    
    # Set the alpha channel to 0 where mask is 1 (person)
    rgba_image[mask] = 0
    
    return rgba_image

def remove_person_and_make_black(image, segmentation_info):
    # Create the segmentation mask
    mask = create_segmentation_mask(image, segmentation_info)
    
    # Set the masked area to black
    image[mask == 255] = 0
    
    return image

def inpaint_image(image, mask):
    inpainted_image = cv2.inpaint(image, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    return inpainted_image

if __name__=='__main__':
    f = open('datasets/train.json')
    data = json.load(f)
    f.close()

    image_dir = 'datasets/yolo_dataset/images/origin/train' # './yolo_dataset/images/train/'
    image_files = os.listdir(image_dir)
    target_path = 'datasets/remove_people'

    if os.path.exists(target_path) is False:
        os.mkdir(target_path)
        print(f"create {target_path} folder.")
    
    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        image = cv2.imread(image_path)
        image_id = int(image_file.split('.')[0])
        segmentation_info = []
        for ann in data['annotations']:
            if ann['image_id'] == image_id:
                person_id = ann['category_id']
                segmentation = ann['segmentation']
                segmentation_info.append({'category_id': person_id, 'segmentation': segmentation})
        #mask = create_segmentation_mask(image, segmentation_info)
        #inpainted_image = inpaint_image(image, mask)
        inpainted_image = remove_person_and_make_black(image, segmentation_info)
        save_name = 're_'+image_file
        cv2.imwrite(f'{os.path.join(target_path,save_name)}', inpainted_image)
        break
    