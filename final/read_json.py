import json


with open("datasets/train.json", "r") as st_json:
    data = json.load(st_json)

print(data.keys())
print(f"#"*30,"image","#"*30)
imgs = data['images']
print(len(imgs))
print(imgs[0])

print(f"15.png image data : {imgs[15]}")

print(f"#"*30,"annotation","#"*30)
anno = data['annotations']

print(len(anno))

print(anno[0].keys())

print(f"id : {anno[0]['id']}")
print(f"image_id : {anno[0]['image_id']}")
print(f"category_id : {anno[0]['category_id']}")
print(f"bbox : {anno[0]['bbox']}")
print(f"area : {anno[0]['area']}")

import os 

print(f" train {len(os.listdir('datasets/yolo_dataset/images/origin/train'))}")
print(f" unlabeld {len(os.listdir('datasets/yolo_dataset/images/origin/un'))}")
print(f" val {len(os.listdir('datasets/yolo_dataset/images/origin/val'))}")
print(f" test {len(os.listdir('datasets/test'))}")
