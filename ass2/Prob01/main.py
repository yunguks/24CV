import torch.nn as nn
from model import SegNet
from PIL import Image
import torchvision
from tqdm import tqdm
from utils import *
import cv2
from torchvision.utils import save_image
import torch.nn.functional as F
from torch.autograd import Variable

class Dataset(object):
    def __init__(self, img_path, label_path, method='train'):
        self.img_path = img_path
        self.label_path = label_path
        self.train_dataset = []
        self.test_dataset = []
        self.mode = method == 'train'
        self.preprocess()
        if self.mode:
            self.num_images = len(self.train_dataset)
        else:
            self.num_images = len(self.test_dataset)

    def preprocess(self):
        for i in range(len([name for name in os.listdir(self.img_path) if os.path.isfile(os.path.join(self.img_path, name))])):
            img_path = os.path.join(self.img_path, str(i) + '.jpg')
            label_path = os.path.join(self.label_path, str(i) + '.png')
            if self.mode == True:
                self.train_dataset.append([img_path, label_path])
            else:
                self.test_dataset.append([img_path, label_path])

        print('Finished preprocessing the CelebA dataset...')

    def __getitem__(self, index):
        dataset = self.train_dataset if self.mode == True else self.test_dataset
        img_path, label_path = dataset[index]
        image = Image.open(img_path)
        label = Image.open(label_path)
        transform = torchvision.transforms.Compose(
            [torchvision.transforms.Resize((512, 512)), torchvision.transforms.ToTensor()])
        label_transform = torchvision.transforms.Compose(
            [torchvision.transforms.Resize((512,512)), torchvision.transforms.PILToTensor()]
        )
        
        return transform(image), label_transform(label).long(), img_path.split("/")[-1]

    def __len__(self):
        """Return the number of images."""
        return self.num_images


class Tester(object):
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.model = self.build_model()
        # Load of pretrained_weight file
        weight_PATH = 'final_model.pth'
        self.model.load_state_dict(torch.load(weight_PATH))
        dataset = Dataset(img_path="data/test_img", label_path="data/test_label", method='test')
        self.dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                                      batch_size=self.batch_size,
                                                      shuffle=True,
                                                      num_workers=2,
                                                      drop_last=False)
        print("Testing...")

    def test(self):
        make_folder("test_mask", '')
        make_folder("test_color_mask", '')
        self.model.eval()
        for i, data in enumerate(self.dataloader):
            imgs = data[0].cuda()
            labels_predict = self.model(imgs)
            labels_predict_plain = generate_label_plain(labels_predict, 512)
            labels_predict_color = generate_label(labels_predict, 512)
            batch_size = labels_predict.size()[0]
            for k in range(batch_size):
                cv2.imwrite(os.path.join("test_mask", data[2][k]), labels_predict_plain[k])
                save_image(labels_predict_color[k], os.path.join("test_color_mask", data[2][k]))

    def build_model(self):
        model = SegNet(3).cuda()
        return model


class Trainer(object):
    def __init__(self, epochs, batch_size, lr):
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = lr
        self.model = self.build_model()
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.learning_rate)
        dataset = Dataset(img_path="data/train_img", label_path="data/train_label", method='train')
        self.dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                                      batch_size=self.batch_size,
                                                      shuffle=True,
                                                      num_workers=2,
                                                      drop_last=False)
        self.criterion = nn.CrossEntropyLoss()
        
    def train(self):
        self.model.train()
        best_score = 10000
        for epoch in range(self.epochs):
            losses = 0.
            acces = 0
            with tqdm(self.dataloader, unit="batch") as pbar:
                for imgs, labels, paths in pbar:
                    pbar.set_description(f"Epoch {epoch+1}")
                    
                    imgs = imgs.cuda()
                    labels = labels.cuda()
                    
                    self.optimizer.zero_grad()
                    preds = self.model(imgs)
                    
                    loss = self.criterion(preds, labels.squeeze(1))
                    
                    loss.backward()
                    self.optimizer.step()
                    
                    losses += loss.item()
                    acc = self._pixel_accuracy(preds, labels)
                    acces += acc
                
                    pbar.set_postfix({
                        'acc' :acc,
                        'loss': loss.item()})
                    
                losses /= len(self.dataloader)
                acces /= len(self.dataloader)
                
                if best_score > losses:
                    best_score = losses
                    torch.save(self.model.eval().state_dict(), 'final_model.pth')
                    print("save best model")
                
    def build_model(self):
        model = SegNet(3)
        # weight_PATH = 'pretrained_weight.pth'
        # model.load_state_dict(torch.load(weight_PATH))
        return model.cuda()

    def _pixel_accuracy(self, output, mask):
        with torch.no_grad():
            output = torch.argmax(F.softmax(output, dim=1), dim=1)
            correct = torch.eq(output, mask).int()
            accuracy = float(correct.sum()) / float(correct.numel())
        return accuracy

if __name__ == '__main__':
    epochs = 10
    lr = 0.0001
    batch_size = 32
    trainer = Trainer(epochs, batch_size, lr)
    trainer.train()
    tester = Tester(32)
    tester.test()
