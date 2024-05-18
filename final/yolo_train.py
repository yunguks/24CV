from ultralytics import YOLO
import os
import sys

os.environ['WANDB__EXECUTABLE']=sys.executable
# Load a model
# model = YOLO("yolov8n.yaml")  # build a new model from scratch
model = YOLO("yolov8x.pt")  # load a pretrained model (recommended for training)

# Use the model
model.train(data="cfg/data.yaml", cfg= 'cfg/custom.yaml')  # train the model
# metrics = model.val()  # evaluate model performance on the validation set