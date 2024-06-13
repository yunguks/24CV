from ultralytics import YOLO
import os
import sys
os.environ['WANDB__EXECUTABLE']=sys.executable

# Load a model
model = YOLO("simclr.pt", task='detect')  # 
# model = YOLO("runs/detect/train_8n/weights/best.pt", task='detect')  # load a pretrained model (recommended for training)

# Use the model
model.train(cfg="cfg/custom.yaml")  # train the model
