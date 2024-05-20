from ultralytics import YOLO
import os
import sys
from sam.trainer import SAMDetectionTrainer

os.environ['WANDB__EXECUTABLE']=sys.executable
# Load a model
# model = YOLO("yolov8n.yaml")  # build a new model from scratch
model = YOLO("yolov8n.pt",task='detect')  # load a pretrained model (recommended for training)

# Use the model
# if using SAM
# model.train(trainer = SAMDetectionTrainer, data="cfg/data.yaml", cfg= 'cfg/custom.yaml')  # train the model

model.train(data="cfg/data.yaml", cfg= 'cfg/custom.yaml')  # train the model
# metrics = model.val()  # evaluate model performance on the validation set