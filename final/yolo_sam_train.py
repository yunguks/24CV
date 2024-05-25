from sam.load_yolo import YOLO
import os
import sys

os.environ['WANDB__EXECUTABLE']=sys.executable
# Load a model
model = YOLO("yolov8x.pt", task='detect')  # 
# model = YOLO("runs/detect/train_8n/weights/best.pt", task='detect')  # load a pretrained model (recommended for training)

# Use the model
# if using SAM
model.train(cfg="cfg/sam.yaml")  # train the model
