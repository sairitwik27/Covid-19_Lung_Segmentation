import os
from PIL import Image
from torch.utils.data import Dataset,ConcatDataset
from torchvision import transforms

class BinaryCovid(Dataset):
    def __init__(self,image_root,gt_root,trainsize):
        self.trainsize = trainsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts =  [gt_root + f for f in os.listdir(gt_root) if f.endswith('.png')]
        self.images=sorted(self.images)
        self.gts=sorted(self.gts)
        self.filter_files()
        self.size = len(self.images)
        
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])


    def __getitem__(self,index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        image_tr = self.img_transform(image)
        gt_tr = self.gt_transform(gt)
        #image_concat = ConcatDataset([image_tr,image])
        #gt_concat = ConcatDataset([gt_tr,gt])
        return image_tr,gt_tr
    
    def filter_files(self):
        assert len(self.images) == len(self.gts)
        images = []
        gts = []
        for img_path, gt_path in zip(self.images, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
        self.images = images
        self.gts = gts
        
    def rgb_loader(self,path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')
            
    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')
    def resize(self, img, gt):
        assert img.size == gt.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST)
        else:
            return img, gt
    def __len__(self):
        return self.size
            
