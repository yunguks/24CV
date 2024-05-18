import torch
import torchvision
from torchvision import transforms
import torch.utils.data as data
import os
from PIL import Image

import torch.nn as nn
from tqdm import tqdm
import numpy as np
import random

def manual_seed(seed):
    np.random.seed(seed) #1
    random.seed(seed) #2
    torch.manual_seed(seed) #3
    torch.cuda.manual_seed(seed) #4.1
    torch.cuda.manual_seed_all(seed) #4.2
    torch.backends.cudnn.benchmark = False #5 
    torch.backends.cudnn.deterministic = True #6
    
class ClockDataset(data.Dataset):
    SHAPE = {
        'circle':0,
        'rectangle':1
        }
    COLOR = {
        'Green':0,
        'Blue':1,
        'Red':2,
        'Yellow':3,
        'Purple':4,
        'Orange':5,
        }
    APM = {
        'AM' :0,
        'PM' :1,
    }
    
    TRANSFROM = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5) ##  mean, std
    ])
    
    
    def __init__(self, data_path ="./train", transform=None):
        
        self.data_path = data_path
        self.file_list = os.listdir(data_path)
        
        if transform:
            self.transform = transform
        else:
            self.transform = self.TRANSFROM
        
        self.imgs = []
        # shape, color, A/PM, hour, minutes
        self.labels = []
        
        for file in self.file_list:
            f = os.path.join(self.data_path, file)
            img = Image.open(f)
            
            shape, color, apm, hour, minute, _ = file.split("_")
            
            label = torch.tensor((self.SHAPE[shape], self.COLOR[color], self.APM[apm], int(hour), int(minute)))
            
            self.imgs.append(img)
            self.labels.append(label)
            
    
    def __getitem__(self, index):
        
        img = self.imgs[index]
        label = self.labels[index]
        
        if self.transform:
            img = self.transform(img)
        
        
        return img, label

    def __len__(self):
        return len(self.labels)
    
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        
        self.backbone = torchvision.models.resnet18(
            weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1
            )

        # # backborn freeze
        # for name, param in self.backbone.named_parameters():
        #     param.requires_grad = False
    
        self.shape_classifier = nn.Linear(1000,2)
        self.color_classifier = nn.Linear(1000,6)
        self.apm_classifier = nn.Sequential(
            nn.Linear(1000,100),
            nn.Linear(100,2),
            )
        self.hour_classifier = nn.Sequential(
            nn.Linear(1000,100),
            nn.Linear(100,12),
            )
        self.minute_classifier = nn.Sequential(
            nn.Linear(1000,200),
            nn.Linear(200,60),
            )
        
    def forward(self, x):
        x = self.backbone(x)
        
        shape = self.shape_classifier(x)
        color = self.color_classifier(x)
        apm = self.apm_classifier(x)
        hour = self.hour_classifier(x)
        minute = self.minute_classifier(x)
        
        output = {
            'shape' : shape,
            'color' : color,
            'apm' : apm,
            'hour' : hour,
            'minute' : minute
        }
        
        return output
    
class ShapeModel(nn.Module):
    def __init__(self):
        super(ShapeModel, self).__init__()
        
        self.backbone = torchvision.models.resnet18(
            weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1
            )

        # backborn freeze
        # for name, param in self.backbone.named_parameters():
        #     param.requires_grad = False
    
        self.shape_classifier = nn.Linear(1000,2)
        self.color_classifier = nn.Linear(1000,6)
        
    def forward(self, x):
        x = self.backbone(x)
        
        shape = self.shape_classifier(x)
        color = self.color_classifier(x)
        
        return [shape, color]
    
class TimeModel(nn.Module):
    def __init__(self):
        super(TimeModel, self).__init__()
        
        self.backbone = torchvision.models.resnet18(
            weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1
            )

        # backborn freeze
        # for name, param in self.backbone.named_parameters():
        #     param.requires_grad = False
    
        self.apm_classifier = nn.Sequential(
            nn.Linear(1000,100),
            nn.Linear(100,2),
            )
        self.hour_classifier = nn.Sequential(
            nn.Linear(1000,100),
            nn.Linear(100,12),
            )
        self.minute_classifier = nn.Sequential(
            nn.Linear(1000,200),
            nn.Linear(200,60),
            )
        
    def forward(self, x):
        x = self.backbone(x)
        
        apm = self.apm_classifier(x)
        hour = self.hour_classifier(x)
        minute = self.minute_classifier(x)
        
        return [apm, hour, minute]
    
class CombineModel(nn.Module):
    def __init__(self,model1, model2):
        super(CombineModel,self).__init__()
        
        self.model1 = model1
        self.model2 = model2

    def forward(self,x):
        shape, color = self.model1(x)
        
        apm, hour, minute = self.model2(x)
        
        output = {
            'shape' : shape,
            'color' : color,
            'apm' : apm,
            'hour' : hour,
            'minute' : minute
        }
        
        return output
    
def convert_output(output):
    shape_dict = {
        0 : 'circle',
        1 : 'rectangle'
    }
    
    color_dict = {
        0 : 'green',
        1 : 'blue',
        2 : 'red',
        3 : 'yellow',
        4 : 'purple',
        5 : 'orange'
    }
    
    apm_dict = {
        0 : 'am',
        1 : 'pm'
    }
    for key in output.keys():
        new_value = []
        for v in output[key]:
            if key == 'shape':
                v = shape_dict[v]
            elif key == 'color':
                v = color_dict[v]
            elif key == 'apm':
                v = apm_dict[v]
            else:
                pass
            new_value.append(v)
        output[key] = new_value
    return output

if __name__ == "__main__":
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    
    # data
    img_path = 'val/circle_Blue_AM_0_40_0.png'
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5) ##  mean, std
    ])
    # image transform
    img = Image.open(img_path)
    img = transform(img).unsqueeze(0)
    
    # model
    model = Model()
    
    # weight load
    model_state_dict = torch.load('./checkpoint/best.pth',map_location='cpu')['model_state_dict']
    model.load_state_dict(model_state_dict)
    
    # # 모양과 시간 예측하는 모델 따로 만들었을 경우 사용
    # model_state_dict = torch.load('./checkpoint/worst.pth',map_location='cpu')
    # shape_model = model_state_dict['shape_model']
    # time_model = model_state_dict['time_model']
    
    # shape_model.load_state_dict(model_state_dict['shape_model_state_dict'])
    # time_model.load_state_dict(model_state_dict['time_model_state_dict'])
    
    # model = CombineModel(shape_model, time_model)
    
    
    ## predict
    model.to(device)
    model.eval()
    with torch.no_grad():
        preds = {}
        img = img.to(device)
        outputs = model(img)
        
        # dictionary 각 키 마다 output 변환
        for i,key in enumerate(outputs.keys()):
            _, pred = torch.max(outputs[key], 1)
            preds[key] = pred.detach().cpu().numpy()
            
        preds = convert_output(preds)
        print(f"Predition {preds}")
    
    # ## for test dataset
    # test_dataset = ClockDataset(data_path='./test')
    # test_loader = torch.utils.data.DataLoader(test_dataset, 128, shuffle=False)
    # model.eval()
    # with torch.no_grad():
    #     val_acc = {'shape': 0, 'color': 0, 'apm': 0, 'hour': 0, 'minute' : 0}
    #     val_progress_bar = tqdm(enumerate(test_loader), total=len(test_loader), desc="Val step", position=1, leave=False)
    #     for step, (imgs, labels) in val_progress_bar:
    #         imgs = imgs.to(device)
    #         labels = labels.transpose(1,0)
    #         labels = labels.to(device)
            
    #         outputs = model(imgs)
            
    #         for i,key in enumerate(val_acc.keys()):
    #             _, preds = torch.max(outputs[key], 1)
    #             val_acc[key] += (preds==labels[i]).sum().item()
        
    #     # accuracy 측정        
    #     for i,key in enumerate(val_acc.keys()):
    #         val_acc[key] = val_acc[key] * 100/ len(test_dataset)
    #         print(f"{key} : {val_acc[key]:.2f}%",end=', ')
    #     print()
        
    #     total_acc = sum(val_acc.values()) / len(val_acc)
    #     print(f'total acc : {total_acc:.2f}')
        