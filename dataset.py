import os
import cv2
from PIL import Image
import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import matplotlib.pyplot as plt

from utils import one_hot_to_image, image_to_class_index


class SatelliteImageDataset(Dataset):
    def __init__(self, image_dir, mask_dir, num_classes=6):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.num_classes = num_classes

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

        self.mask_labels = ['building', 'land', 'road', 'vegetation', 'water', 'unlabeled']

        assert len(self.image_filenames) == len(self.mask_filenames)

    def __getitem__(self, i):
        image_path = os.path.join(self.image_dir, self.image_filenames[i])
        mask_path = os.path.join(self.mask_dir, self.mask_filenames[i])

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

        transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((512, 512)),
                transforms.ColorJitter(contrast=(1.25, 1.25)),
                transforms.ToTensor(),
            ])
        # (c, h, w)
        image = transform(image)
        sharpened_image = transforms.functional.adjust_sharpness(image, 2)
        
        resized_mask = cv2.resize(mask, (512, 512))
        # (c, h, w)
        mask = torch.tensor(resized_mask).permute(2, 0, 1)

        mask = image_to_class_index(mask, self.color_map)
        mask = mask.long()

        return sharpened_image, mask

    def __len__(self):
        return len(self.image_filenames)
    

if __name__ == '__main__':
    transform = transforms.Compose([
        #transforms.ToPILImage(),
        transforms.Resize((512, 512)),
    ])

    dataset = SatelliteImageDataset(
        image_dir='data/images',
        mask_dir='data/masks',
        transform=transform
    )

    img, mask = dataset[0]
    print(img.shape)
    print(mask.shape)
    mask = torch.tensor(one_hot_to_image(mask, dataset.color_map))
    print(mask.shape)
    plt.imshow(mask.permute(1, 2, 0))

    plt.show()
    """for i in range(6):
        plt.imshow(mask[i], cmap='gray')
        plt.show()"""

    print(mask.min(), mask.max())

