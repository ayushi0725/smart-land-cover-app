import os

import cv2
from PIL import Image
import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torchvision.transforms import v2 as T
import matplotlib.pyplot as plt
import numpy as np

from utils import one_hot_to_image, image_to_class_index, class_index_to_image


class SatelliteImageDataset(Dataset):
    def __init__(self, image_dir, mask_dir, dest_image_dir=None, dest_mask_dir=None, num_classes=6):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.augmented_image_dir = dest_image_dir
        self.augmented_mask_dir = dest_mask_dir
        self.num_classes = num_classes

        self.image_filenames = sorted(os.listdir(self.image_dir))
        self.mask_filenames = sorted(os.listdir(self.mask_dir))

        if dest_image_dir and dest_mask_dir:
            self.augmented_image_filenames = sorted(os.listdir(self.augmented_image_dir))
            self.augmented_mask_filenames = sorted(os.listdir(self.augmented_mask_dir))
        else:
            self.augmented_image_filenames  = []
            self.augmented_mask_filenames = []

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
        assert len(self.augmented_image_filenames) == len(self.augmented_mask_filenames)

    def __getitem__(self, i):
        if self.augmented_image_dir and self.augmented_mask_dir:
            image_path = os.path.join(self.augmented_image_dir, self.augmented_image_filenames[i])
            mask_path = os.path.join(self.augmented_mask_dir, self.augmented_mask_filenames[i])
        else:
            image_path = os.path.join(self.image_dir, self.image_filenames[i])
            mask_path = os.path.join(self.mask_dir, self.mask_filenames[i])

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ColorJitter(contrast=(1.25, 1.25)),
            transforms.Resize((512, 512)),
            transforms.ToTensor()
        ])
        # (c, h, w)
        image = transform(image)
        sharpened_image = transforms.functional.adjust_sharpness(image, 2)
        
        resized_mask = cv2.resize(mask, (512, 512))
        # (c, h, w)
        mask = torch.tensor(resized_mask, dtype=torch.uint8).permute(2, 0, 1)

        mask = image_to_class_index(mask, self.color_map)
        mask = mask.long()
        
        return sharpened_image, mask

    def __len__(self):
        return len(self.augmented_image_filenames) if self.augmented_image_filenames else len(self.image_filenames)
    
    def augmentation(self, dest_image_dir, dest_mask_dir):
        self.augmented_image_dir = dest_image_dir
        self.augmented_mask_dir = dest_mask_dir

        for i in range(5):
            for image_filename, mask_filename in zip(self.image_filenames, self.mask_filenames):
                image = Image.open(os.path.join(self.image_dir, image_filename))
                mask = Image.open(os.path.join(self.mask_dir, mask_filename))

                if i == 0:
                    # Save the original image in the 1st iter
                    transformed_image, transformed_mask = image, mask
                else:
                    transform = T.Compose([
                        T.RandomResizedCrop(size=(512, 512), scale=(1.2, 2)),
                        T.RandomHorizontalFlip(),
                        T.RandomVerticalFlip()
                    ])
                    transformed_image, transformed_mask = transform(image, mask)
        
                    transformed_image = transforms.ColorJitter(
                        brightness=(0.8, 1.3), hue=(-0.15, 0.15)
                    )(transformed_image)

                # Get the filename of the new image (name + version)
                transformed_image_filename = image_filename[:-4] + f'v{i}'
                transformed_mask_filename = mask_filename[:-4] + f'v{i}'

                # Get the full path of the image
                image_path = os.path.join(self.augmented_image_dir, transformed_image_filename)
                mask_path = os.path.join(self.augmented_mask_dir, transformed_mask_filename)

                transformed_image.save(image_path + '.jpg')
                transformed_mask.save(mask_path + '.png')

        self.augmented_image_filenames = sorted(os.listdir(self.augmented_image_dir))
        self.augmented_mask_filenames = sorted(os.listdir(self.augmented_mask_dir))
                
    
if __name__ == '__main__':
    transform = transforms.Compose([
        #transforms.ToPILImage(),
        transforms.Resize((512, 512)),
    ])

    dataset = SatelliteImageDataset(
        image_dir='data/images',
        mask_dir='data/masks',
    )

    img, mask = dataset[0]
    print(img.shape)
    print(mask.shape)
    mask = torch.tensor(class_index_to_image(mask.unsqueeze(0), dataset.color_map))
    print(mask.shape)
    #plt.imshow(mask.squeeze(0).permute(1, 2, 0))
    plt.imshow(img.permute(1, 2, 0))

    plt.savefig("hi.png")
    """for i in range(6):
        plt.imshow(mask[i], cmap='gray')
        plt.show()"""

    print(mask.min(), mask.max())

