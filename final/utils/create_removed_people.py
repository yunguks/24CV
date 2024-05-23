import numpy as np
import cv2
import json
import os
import random

def create_segmentation_mask(image_shape, removing_info):
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    for obj in removing_info:
        polygon = np.array(obj['segmentation'], dtype=np.int32)
        cv2.fillPoly(mask, [polygon], 255)
    return mask

def remove_person_and_fill_background(image, mask, background_image):
    background_image = cv2.resize(background_image, (image.shape[1], image.shape[0]))
    image[mask == 255] = background_image[mask == 255]
    return image

def check_folder(path):
    if os.path.exists(path) is False:
        os.mkdir(path)
        print(f"create {path} folder.")

def select_background(image_id):
    # 첫번째 배경: 0 ~ 61,  기준 25 / 29 사용
    # 두번째 배경: 124~179, 기준 132 / 129 사용
    # 세번째 배경: 237~283, 기준 243 / 270 사용
    # 네번째 배경: 332~393, 기준 357 / 376 사용
    if image_id < 62:
        return 0
    elif image_id < 180:
        return 1
    elif image_id < 284:
        return 2
    else:
        return 3

if __name__=='__main__':
# Load the annotations
    with open('datasets/train.json') as f:
        data = json.load(f)

    # Directory containing images
    image_dir = 'datasets/yolo_dataset/images/origin/train'
    image_files = os.listdir(image_dir)

    target_dir = 'datasets/remove_people'
    target_image_dir = os.path.join(target_dir,'images')
    target_label_dir = os.path.join(target_dir,'labels')
    check_folder(target_dir)
    check_folder(target_image_dir)
    check_folder(target_label_dir)
    
    background_images = []
    background_images.append(cv2.imread(os.path.join(image_dir,'b1.png')))
    background_images.append(cv2.imread(os.path.join(image_dir,'b2.png')))
    background_images.append(cv2.imread(os.path.join(image_dir,'b3.png')))
    background_images.append(cv2.imread(os.path.join(image_dir,'b4.png')))
    
    # Process each image
    W = 640
    H = 345
    
    # label이 있는 전체 파일 대상
    for image_file in image_files:
        if 'b' in image_file:
            continue
        print(f"{image_file} : ", end='')
        image_path = os.path.join(image_dir, image_file)
        image = cv2.imread(image_path)
        image_id = int(os.path.splitext(image_file)[0])  # Assuming image file name is the image_id
        segmentation_info = []

        W, H = data['images'][image_id]['width'], data['images'][image_id]['height']
        save_person_id = []
        
        for ann in data['annotations']:
            if ann['image_id'] == image_id:
                person_id = ann['category_id']
                save_person_id.append(person_id)
                segmentation = ann['segmentation']
                bbox = ann['bbox']
                segmentation_info.append({'category_id': person_id, 'segmentation': segmentation, 'bbox': bbox})
        
        # save_person_id 중복제거
        save_person_id = list(set(save_person_id))
        
        # box 중 무조껀 1개 이상 제거
        if len(save_person_id) > 1:
            num_segments = random.randint(1, len(save_person_id)-1)  # Ensure at least one segment is selected
        else:
            num_segments = 1
        
        # 배경선택
        background_id = select_background(image_id)
        
        # 제거할 사람 id 선택
        removing_id = random.sample(save_person_id, num_segments)
        print(f"{removing_id} person removed")
        removing_info = [info for info in segmentation_info if info['category_id'] in removing_id]
        
        # 제거한 후 info
        remaining_info = [info for info in segmentation_info if info['category_id'] not in removing_id]

        # 사람 제거, 제거할 segment 정보 
        mask = create_segmentation_mask(image.shape, removing_info)
        
        # 배경 입히기
        result_image = remove_person_and_fill_background(image, mask, background_images[background_id])
        
        # 저장
        save_name = 're_'+image_file
        output_path = os.path.join(target_image_dir,save_name)
        cv2.imwrite(output_path, result_image)
        
        label_path = os.path.join(target_label_dir,save_name.replace('png','txt'))
        with open(label_path, 'w') as f:
            for info in remaining_info:
                category_id = info['category_id']
                bbox = info['bbox']
                x = (bbox[0]+bbox[2]/2) / W
                y = (bbox[1]+bbox[3]/2) / H
                w = bbox[2] / W
                h = bbox[3] / H
                
                f.write(f'{category_id} {x} {y} {w} {h}\n')
