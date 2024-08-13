import os
import cv2
import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import matplotlib.pyplot as plt

from utils import image_to_one_hot


class SatelliteImageDataset(Dataset):
    def __init__(self, image_dir, mask_dir, num_classes=6, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.num_classes = num_classes
        self.transform = transform

        self.image_filenames = sorted(os.listdir(self.image_dir))
        self.mask_filenames = sorted(os.listdir(self.mask_dir))

        self.color_map = {
            (60, 16, 152): 0, # building
            (132, 41, 246): 1, # land
            (110, 193, 228): 2, # road
            (254, 221, 58): 3, # vegetation
            (226, 169, 41): 4, # water
            (155, 155, 155): 5 # unlabeled / unknown
        }

        assert len(self.image_filenames) == len(self.mask_filenames)

    def __getitem__(self, i):
        image_path = os.path.join(self.image_dir, self.image_filenames[i])
        mask_path = os.path.join(self.mask_dir, self.mask_filenames[i])

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

        plt.imshow(mask)
        plt.show()
            
        # Convert the mask into one hot encoding with 1 channel for each class
        mask = image_to_one_hot(mask, self.num_classes, self.color_map)
        # (h, w, c)
        mask = torch.tensor(mask, dtype=torch.float32).permute(1, 2, 0)

        if self.transform:
            image = self.transform(image)
            resized_mask = cv2.resize(mask.numpy(), (512, 512))
            # (c, h, w)
            mask = torch.tensor(resized_mask).permute(2, 0, 1)

        return image, mask

    def __len__(self):
        return len(self.image_filenames)
    

if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])

    dataset = SatelliteImageDataset(
        image_dir='data/images',
        mask_dir='data/masks',
        transform=transform
    )

    img, mask = dataset[0]
    print(img.shape)
    print(mask.shape)

    plt.imshow(img.permute(1, 2, 0))
    plt.show()
    for i in range(6):
        plt.imshow(mask[i], cmap='gray')
        plt.show()

    print(mask.min(), mask.max())