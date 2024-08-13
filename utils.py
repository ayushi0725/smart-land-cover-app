import numpy as np


def crop(img, to_dim):
    h, w = to_dim.shape[2:]
    img_h, img_w = img.shape[2:]
    
    crop_h = (img_h - h) // 2
    crop_w = (img_w - w) // 2
    
    return img[:, :, crop_h: crop_h + h, crop_w: crop_w + w]
    

def image_to_one_hot(img, num_classes, color_map):
    h, w = img.shape[:2]
    one_hot_mask = np.zeros((num_classes, h, w))

    for color, color_map_idx in color_map.items():
        mask_indices = np.all(img == np.array(color).reshape(1, 1, 3), axis=-1)
        one_hot_mask[color_map_idx, mask_indices] = 1

    return one_hot_mask