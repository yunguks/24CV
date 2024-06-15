import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import glob
import os
from torchvision.utils import save_image
import pickle

def convert_pickle(root):
    img_size = 128
    transforms_ = [transforms.Resize((img_size, img_size), Image.BICUBIC), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    transform = transforms.Compose(transforms_)
    root = root
    files = sorted(glob.glob("%s/*.jpg" % root))
    
    target = root.replace('train','resize')
    
    if not os.path.exists(target):
        os.mkdir(target)
    
    
    for file_name in files:
        img = Image.open(file_name)

        img = transform(img)
        
        target_name = file_name.replace('train', 'resize')
        pickle_path = target_name.replace('jpg','.pkl')
        with open(pickle_path, 'wb') as f:
            pickle.dump(img, f)
        
    print("Done")
    
    
if __name__=='__main__':
    convert_pickle('data/train')