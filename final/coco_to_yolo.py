
import json
import cv2
import os
import matplotlib.pyplot as plt
import shutil
import random

input_path = "./datasets/"
output_path = "./datasets/yolo_dataset/"

f = open('./datasets/train.json')
data = json.load(f)
f.close()

file_names = []
if os.path.exists(output_path)==False:
    os.mkdir(output_path)

for p_0 in ['images','labels']:
    up_f = os.path.join(output_path,p_0)
    if os.path.exists(up_f)==False:
        os.mkdir(up_f)
    
    for p_1 in ['train','val','un']:
        mkidr_path = os.path.join(up_f,p_1)
        if os.path.exists(mkidr_path)==False:
            os.mkdir(mkidr_path)


def load_images_from_folder(folder):
  for filename in sorted(os.listdir(folder)):
    source = os.path.join(folder,filename)
    destination = f"{output_path}/images/train/{filename}"

    try:
        shutil.copy(source, destination)
    # If source and destination are same
    except shutil.SameFileError:
        print("Source and destination represents the same file.")

    file_names.append(os.path.join('train',filename))

load_images_from_folder('./datasets/train')

def get_img_ann(image_id):
    img_ann = []
    isFound = False
    # print(image_id)
    for ann in data['annotations']:
        if ann['image_id'] == image_id:
            # print(ann['image_id'])
            img_ann.append(ann)
            isFound = True
    if isFound:
        return img_ann
    else:
        return None
    
    
def get_img(filename):
  for img in data['images']:
    if img['file_name'] == filename:
      return img
    
    

for filename in file_names:
  # Extracting image 
  img = get_img(filename)
  img_id = img['id']
  img_w = img['width']
  img_h = img['height']

  # Get Annotations for this image
  img_ann = get_img_ann(img_id)

  if img_ann:
    # Opening file for current image
    file_object = open(f"{output_path}labels/train/{img_id}.txt", "a")

    for ann in img_ann:
      current_category = ann['category_id'] # As yolo format labels start from 0 
      current_bbox = ann['bbox']
      x = current_bbox[0]
      y = current_bbox[1]
      w = current_bbox[2]
      h = current_bbox[3]
      
      # Finding midpoints
      x_centre = x + w/2
      y_centre = y + h/2
      
      # Normalization
      x_centre = x_centre / img_w
      y_centre = y_centre / img_h
      w = w / img_w
      h = h / img_h
      
      # Limiting upto fix number of decimal places
      x_centre = format(x_centre, '.6f')
      y_centre = format(y_centre, '.6f')
      w = format(w, '.6f')
      h = format(h, '.6f')
          
      # Writing current object 
      file_object.write(f"{current_category} {x_centre} {y_centre} {w} {h}\n")

    file_object.close()
    
    
def split_images_from_folder(folder):
  
    train_img_folder = os.path.join(folder,'images','train')
    val_img_folder = os.path.join(folder,'images','val')
    
    train_label_folder = train_img_folder.replace('images','labels')
    val_label_folder = val_img_folder.replace('images','labels')
    
    unlabel_img_folder = os.path.join(folder,'images','un')
    
    for filename in os.listdir(train_img_folder):
        source = os.path.join(train_img_folder,filename)
        
        label_name = filename.replace('png', 'txt')
        source_label = os.path.join(train_label_folder, label_name)
        
        if os.path.exists(source_label):
            if random.random() < 0.2:
                destination = os.path.join(val_img_folder,filename)
                destination_label = os.path.join(val_label_folder,label_name)

                try:
                    print(source,"->", destination)
                    print(source_label,"->", destination_label)
                    shutil.move(source, destination)
                    shutil.move(source_label, destination_label)
                # If source and destination are same
                except shutil.SameFileError:
                    print("Source and destination represents the same file.")
        
        else:
            destination = os.path.join(unlabel_img_folder,filename)
            shutil.move(source, destination)
            print("move unlabeled folder")


split_images_from_folder('./datasets/yolo_dataset')