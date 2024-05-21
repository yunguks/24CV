from ultralytics import YOLO
import os
import sys
from sam.trainer import SAMDetectionTrainer
from ultralytics.models.yolo.detect import DetectionTrainer

os.environ['WANDB__EXECUTABLE']=sys.executable
# Load a model
# model = YOLO("yolov8n.pt")  # 
model = YOLO("runs/detect/train_8n/weights/best.pt", task='detect')  # load a pretrained model (recommended for training)

# Use the model
# if using SAM
model.train(trainer = SAMDetectionTrainer, cfg="cfg/custom.yaml")  # train the model

# model.train(cfg= 'cfg/custom.yaml')  # train the model
# metrics = model.val()  # evaluate model performance on the validation set