from ultralytics import YOLO
import os
import sys
import time
from PIL import Image
import torchvision
import json
import numpy as np
import zipfile

os.environ['WANDB__EXECUTABLE']=sys.executable
# Load a model
# model = YOLO("yolov8n.yaml")  # build a new model from scratch
model = YOLO("runs/detect/test4/weights/best.pt").cuda()  # load a pretrained model (recommended for training)

# # Use the model
img = Image.open('datasets/yolo_dataset/images/val/6.png')
transform = torchvision.transforms.Compose([
    # torchvision.transforms.Resize((448,448)),
    torchvision.transforms.Resize((640,640)),
    torchvision.transforms.ToTensor(),
    # torchvision.transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])
img = transform(img)
img = img.to('cuda').unsqueeze(0)
# print('warmup..')
# for i in range(100):
#     source = torch.rand(1, 3, 448, 448, dtype=torch.float32).to('cuda')
#     result = model(source)
    
start = time.time()
result = model(img)  # evaluate model performance on the validation set
print(result[0].boxes)
sec = time.time()-start

print(f"{sec:.2f}ms")

test_path = 'datasets/test'

for fname in os.listdir(test_path):
    img_path = os.path.join(test_path,fname)
    
    img = Image.open(img_path)
    img = transform(img)
    img = img.to('cuda').unsqueeze(0)
    
    result = model(img)[0]
    
    boxes = result.boxes
    
    pred = []
    for i in range(len(boxes.cls)):
        result_dict= {
            'image_id' : int(fname.replace('.png','')),
            'category_id' : int(boxes.cls[i]),
            'bbox' : np.array(boxes.xywh[i].detach().cpu(),dtype=np.int32).tolist(),
            'score' : float(boxes.conf[i])
        }
        pred.append(result_dict)
        
print(pred[0])

with open("pred.json", "w") as json_file:
    json.dump(pred, json_file)
    
# Create a ZIP file and add the JSON file to it
zip_file = zipfile.ZipFile("predictions.zip", "w")
zip_file.write("pred.json")
zip_file.close()
