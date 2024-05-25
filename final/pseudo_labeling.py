from ultralytics import YOLO
import os
import torch
import cv2
import argparse 


what_type = 'un'

model = YOLO('./runs/detect/sam_8x_background/weights/best.pt')

unlabeled_data_dir = './datasets/yolo_dataset/images/origin/un'
pseudo_labels_dir = unlabeled_data_dir.replace('images','labels')

def convert_segmentation_to_yolo_format(segmentation, img_w, img_h):
    segmentation_yolo = []
    for point in segmentation:
        x, y = point
        segmentation_yolo.append(x / img_w)
        segmentation_yolo.append(y / img_h)
    return segmentation_yolo

unlabeled_images = [os.path.join(unlabeled_data_dir, img) for img in os.listdir(unlabeled_data_dir)]
for img_path in unlabeled_images:
    results = model(img_path)
    # 예측 결과를 가짜 라벨로 저장 (YOLO 포맷으로 저장한다고 가정)
    pseudo_label_path = os.path.join(pseudo_labels_dir, os.path.basename(img_path).replace('.png', '.txt'))
    
    img = cv2.imread(img_path)
    img_h, img_w, _ = img.shape
    
    with open(pseudo_label_path, 'w') as f:
        for result in results:
            for box in result.boxes:
                x_center, y_center, w, h, score, class_id = box.xywhn[0].tolist() + [box.conf.item(), box.cls.item()]
                if score > 0.25:  # 신뢰도가 0.5 이상인 경우만 사용
                    f.write(f"{int(class_id)} {x_center} {y_center} {w} {h}\n")