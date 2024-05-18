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
        
        # 파일 읽고 이름에 따른 정답 추출
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
    
if __name__ == "__main__":
    
    manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    
    # dataset 불러오기
    train_dataset = ClockDataset(data_path="./train")
    print(f"Train data size : {len(train_dataset)}")
    val_dataset = ClockDataset(data_path="./val")
    print(f"Validation data size : {len(val_dataset)}")
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size= 128, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size= 128, shuffle=False)
    
    # model
    model = Model()
    
    lr = 1e-4
    
    # optimizer , backbone 모델의 lr을 10배 작게
    optim_group = []
    for name, module in model.named_children():
        if name in 'backbone':
            optim_group.append({'params': module.parameters(), 'lr': lr*0.1})
        else:
            optim_group.append({'params': module.parameters(), 'lr': lr})
            
    optimizer = torch.optim.AdamW(optim_group)
    
    criterion = nn.CrossEntropyLoss()
    
    # train
    epochs = 10
    model.to(device)
    best_acc = 0.
    for epoch in tqdm(range(epochs), desc="Epoch", position=0, leave=False, total=epochs):
        ## train
        model.train()
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc="Train step", position=1, leave=False)
        for step,(imgs, labels) in progress_bar:
            imgs = imgs.to(device)
            labels = labels.transpose(1,0)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            # outputs => (batch, 5 , ..)
            outputs = model(imgs)
            
            # 각각의 loss를 따로 계산해서 하나로 합치기
            shape_loss = criterion(outputs['shape'], labels[0])
            color_loss = criterion(outputs['color'], labels[1])
            apm_loss = criterion(outputs['apm'], labels[2])
            hour_loss = criterion(outputs['hour'], labels[3])
            minute_loss = criterion(outputs['minute'], labels[4])
            
            loss = shape_loss + color_loss + apm_loss + hour_loss + minute_loss
            progress_bar.set_postfix(
                {
                "loss": f'{loss.item():.4f}', 
                "shape": f'{shape_loss.item():.4f}', 
                "color": f'{color_loss.item():.4f}', 
                "apm": f'{apm_loss.item():.4f}', 
                "hour": f'{hour_loss.item():.4f}', 
                "minute": f'{minute_loss.item():.4f}',
                }
            )
    
            loss.backward()
            
            optimizer.step()
        
        ## evalutate
        model.eval()
        with torch.no_grad():
            val_acc = {'shape': 0, 'color': 0, 'apm': 0, 'hour': 0, 'minute' : 0}
            val_progress_bar = tqdm(enumerate(val_loader), total=len(val_loader), desc="Val step", position=1, leave=False)
            for step, (imgs, labels) in val_progress_bar:
                imgs = imgs.to(device)
                labels = labels.transpose(1,0)
                labels = labels.to(device)
                
                outputs = model(imgs)
                
            
            
                for i,key in enumerate(val_acc.keys()):
                    _, preds = torch.max(outputs[key], 1)
                    val_acc[key] += (preds==labels[i]).sum().item()
            
            # accuracy 측정        
            for i,key in enumerate(val_acc.keys()):
                val_acc[key] = val_acc[key] * 100/ len(val_dataset)
                print(f"{key} : {val_acc[key]:.2f}%",end=', ')
            print()
            
            total_acc = sum(val_acc.values()) / len(val_acc)
            
            if not os.path.exists('./checkpoint'):
                os.mkdir('./checkpoint')
            if best_acc < total_acc:
                best_acc = total_acc
                shape_checkpoint = {
                        'model' : model,
                        'model_state_dict' : model.state_dict(),
                    }
                torch.save(shape_checkpoint, f'./checkpoint/best.pth')
                print(f"save best model.")
                