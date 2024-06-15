from model import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import cv2
import glob
import pandas
# from google.colab.patches import cv2_imshow
from PIL import Image
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.autograd import Variable

device = 'cuda:0'

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)

def calculate_psnr(original, restored):
    mse = F.mse_loss(original, restored)
    if mse == 0:  # MSE가 0인 경우, PSNR은 무한대입니다.
        return torch.float('inf')
    max_pixel = 1.0  # 이미지가 0-1 범위로 정규화
    psnr = 10 * torch.log10(max_pixel ** 2 / mse)
    return psnr

import pickle
class CustomDataset(Dataset):
    def __init__(self, root, img_size=128, mask_size=64, method="train"):
        self.img_size = img_size
        self.mask_size = mask_size
        self.mode = method
        self.root = root
        self.files = sorted(glob.glob("%s/*.pkl" % root))
        self.files = self.files[:-4000] if self.mode == "train" else self.files[-4000:]

    def apply_random_mask(self, img):
        """Randomly masks image"""
        y1, x1 = np.random.randint(0, self.img_size - self.mask_size, 2)
        y2, x2 = y1 + self.mask_size, x1 + self.mask_size
        masked_part = img[:, y1:y2, x1:x2]
        masked_img = img.clone()
        masked_img[:, y1:y2, x1:x2] = 1
        return masked_img, masked_part

    def apply_center_mask(self, img):
        """Mask center part of image"""
        # Get upper-left pixel coordinate
        i = (self.img_size - self.mask_size) // 2
        masked_img = img.clone()
        masked_img[:, i : i + self.mask_size, i : i + self.mask_size] = 1
        return masked_img, i

    def __getitem__(self, index):
        pickle_path = self.files[index % len(self.files)]
        
        with open(pickle_path, 'rb') as f:
            img = pickle.load(f)
        if self.mode == "train":
            # For training data perform random mask
            masked_img, aux = self.apply_random_mask(img)
        else:
            # For test data mask the center of the image
            masked_img, aux = self.apply_center_mask(img)
        return img, masked_img, aux

    def __len__(self):
        return len(self.files)


class FaceDataset(Dataset):
    def __init__(self, root, transforms_=None, img_size=128, mask_size=64, method="train"):
        self.transform = transforms.Compose(transforms_)
        self.img_size = img_size
        self.mask_size = mask_size
        self.mode = method
        self.root = root
        self.files = sorted(glob.glob("%s/*.jpg" % root))
        self.files = self.files[:-4000] if self.mode == "train" else self.files[-4000:]

    def apply_random_mask(self, img):
        """Randomly masks image"""
        y1, x1 = np.random.randint(0, self.img_size - self.mask_size, 2)
        y2, x2 = y1 + self.mask_size, x1 + self.mask_size
        masked_part = img[:, y1:y2, x1:x2]
        masked_img = img.clone()
        masked_img[:, y1:y2, x1:x2] = 1
        return masked_img, masked_part

    def apply_center_mask(self, img):
        """Mask center part of image"""
        # Get upper-left pixel coordinate
        i = (self.img_size - self.mask_size) // 2
        masked_img = img.clone()
        masked_img[:, i : i + self.mask_size, i : i + self.mask_size] = 1
        return masked_img, i

    def __getitem__(self, index):
        img = Image.open(self.files[index % len(self.files)])
        img = self.transform(img)
        if self.mode == "train":
            # For training data perform random mask
            masked_img, aux = self.apply_random_mask(img)
        else:
            # For test data mask the center of the image
            masked_img, aux = self.apply_center_mask(img)
        return img, masked_img, aux

    def __len__(self):
        return len(self.files)


class Trainer(object):
    def __init__(self, epochs, batch_size, lr):
        self._build_model()
        transforms_ = [transforms.Resize((128, 128), Image.BICUBIC), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        dataset = FaceDataset(root='data/train', method='train', transforms_=transforms_)
        # dataset = CustomDataset(root='data/resize', method='train')
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        self.epochs = epochs
        self.batch_size = batch_size
        
        self.optimizerG = torch.optim.AdamW(self.gnet.parameters(), lr=lr)
        self.optimizerD = torch.optim.AdamW(self.dnet.parameters(), lr=lr)
        # self.criterion = nn.CrossEntropyLoss()
        self.criterion = nn.MSELoss()
        
    def train(self):
        self.gnet.train()
        self.dnet.train()
        best_score = 0
        for epoch in range(self.epochs):
            psnrs = 0.
            with tqdm(self.dataloader, unit='batch') as pbar:
                pbar.set_description(f"Epoch {epoch+1}")
                for imgs, mask_imgs, aux in pbar:
                    imgs = imgs.to(device)
                    mask_imgs = mask_imgs.to(device)
                    aux = aux.to(device)
                    
                    # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                    self.dnet.zero_grad()
                    
                    # Forward pass real batch through D
                    output = self.dnet(imgs)
                    real_label = torch.ones_like(output,device=device, dtype=torch.float)

                    # Calculate loss on all-real batch
                    errD_real = self.criterion(output, real_label)
                    # Calculate gradients for D in backward 
                    # pass
                    errD_real.backward()
                    D_x = output.mean().item()
                    
                    ## Train with all-fake batch
                    # Generate fake image batch with G
                    fake = self.gnet(mask_imgs)
    
                    # Classify all fake batch with D
                    output = self.dnet(fake.detach())
                    fake_label = torch.zeros_like(output,device=device, dtype=torch.float)
                    # Calculate D's loss on the all-fake batch
                    errD_fake = self.criterion(output, fake_label)
                    # Calculate the gradients for this batch, accumulated (summed) with previous gradients
                    errD_fake.backward()
                    # Compute error of D as sum over the fake and the real batches
                    errD = errD_real + errD_fake
                    # Update D
                    self.optimizerD.step()
                    
                    
                    # (2) Update G network: maximize log(D(G(z)))
                    self.gnet.zero_grad()
                    # Since we just updated D, perform another forward pass of all-fake batch through D
                    output = self.dnet(fake)
                    fake_label = torch.zeros_like(output,device=device, dtype=torch.float)  # fake labels are real for generator cost
                    
                    ## Base loss
                    # # Calculate G's loss based on this output
                    ## 3) additional reconstruction loss
                    adversarial_loss = self.criterion(output, fake_label)
                    # Calculate G's reconstruction loss (L1 loss between fake and real images)
                    reconstruction_loss = F.l1_loss(fake, aux)
                    # Combine the two losses
                    errG = adversarial_loss + 2 * reconstruction_loss
                    
                    # Calculate gradients for G
                    errG.backward()
                    # Update G
                    self.optimizerG.step()
                    
                    
                    psnr = calculate_psnr(fake, aux)
                    psnrs += psnr.item()
                    
                    pbar.set_postfix({
                        'D loss': errD.item(),
                        'G loss': errG.item(),
                        'Recon':reconstruction_loss.item(),
                        'psnr': psnr.item()
                    })
            
            psnrs = psnrs / len(self.dataloader)
            
            if best_score < psnrs:
                best_score = psnrs
                torch.save(self.gnet.eval().state_dict(), 'model.pth')
                print("save best model")
        
    
    def _build_model(self):
        self.gnet = Generator()
        self.dnet = Discriminator()
        self.gnet.load_state_dict(torch.load('pretrained_weight.pth'))
        # self.gnet.apply(weights_init_normal)
        self.dnet.apply(weights_init_normal)
        
        self.gnet = self.gnet.to(device)
        self.dnet = self.dnet.to(device)
        print("Finish build model.")


class Tester(object):
    def __init__(self, batch_size):
        self._build_model()
        transforms_ = [transforms.Resize((128, 128), Image.BICUBIC), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        dataset = FaceDataset(root='data/test', method='test', transforms_=transforms_)
        self.root = dataset.root        
        self.test_dataloader = DataLoader(dataset, batch_size=6, shuffle=False)

        print("Testing...")

    def _build_model(self):
        gnet = Generator()
        self.gnet = gnet.to(device)
        self.gnet.load_state_dict(torch.load('model.pth')) #Change this path
        self.gnet.eval()
        print('Finish build model.')

    def test(self):
        Tensor = torch.cuda.FloatTensor
        samples, masked_samples, i = next(iter(self.test_dataloader))
        samples = Variable(samples.type(Tensor))
        masked_samples = Variable(masked_samples.type(Tensor))
        i = i[0].item()  # Upper-left coordinate of mask

        # Generate inpainted image
        gen_mask = self.gnet(masked_samples)
        filled_samples = masked_samples.clone()
        filled_samples[:, :, i : i + 64, i : i + 64] = gen_mask

        # Save sample
        sample = torch.cat((masked_samples.data, filled_samples.data, samples.data), -2)
        save_image(sample, "test.png", nrow=6, normalize=True)   #Change this path


def main():

    epochs = 200
    batchSize = 32
    learningRate = 0.0001

    trainer = Trainer(epochs, batchSize, learningRate)
    trainer.train()

    tester = Tester(batchSize)
    tester.test()

if __name__ == '__main__':
    main()
