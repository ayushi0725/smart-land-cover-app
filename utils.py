import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch


def crop(img, to_dim):
    h, w = to_dim.shape[2:]
    img_h, img_w = img.shape[2:]
    
    crop_h = (img_h - h) // 2
    crop_w = (img_w - w) // 2
    
    return img[:, :, crop_h: crop_h + h, crop_w: crop_w + w]
    

def class_index_to_one_hot(img, color_map):
    h, w = img.shape # input (h, w)
    img = img.long()
    one_hot_mask = torch.zeros((len(color_map), h, w))

    one_hot_mask.scatter_(0, img.unsqueeze(0), 1)

    return one_hot_mask # output (c, h, w)


def one_hot_to_image(one_hot_imgs, color_map, device='cpu'):
    b, c, h, w = one_hot_imgs.shape # input (b, 6, h, w)
    img_batch = torch.zeros((b, 3, h, w), dtype=torch.uint8).to(device)

    # reverse the color map
    color_map = {idx: color for color, idx in color_map.items()}

    for i in range(b):
        class_indices = torch.argmax(one_hot_imgs[i], axis=0)

        img = torch.zeros((3, h, w), dtype=torch.uint8).to(device)

        for color_map_idx, color in color_map.items():
            # assign the colors to each color channel        
            for ch in range(3):
                img[ch, color_map_idx == class_indices] = color[ch]

        img_batch[i] = img

    return img_batch # output (b, 3, h, w)


def class_index_to_image(class_index_map, color_map, device='cpu'):
    # batch_class_index_map: (batch, h, w)
    b, h, w = class_index_map.shape
    img_batch = torch.zeros((b, 3, h, w), dtype=torch.uint8).to(device)

    # reverse the color map
    color_map = {v: k for k, v in color_map.items()}

    for class_idx, color in color_map.items():
        mask = class_index_map == class_idx
        for i in range(3):  
            img_batch[:, i, :, :][mask] = color[i]

    return img_batch  # img_batch: (b, c, h, w)


def image_to_class_index(img, color_map):
    h, w = img.shape[1:]  # img: (c, h, w)
    class_index_map = torch.zeros((h, w))

    for color, class_idx in color_map.items():
        color_tensor = torch.tensor(color, dtype=img.dtype).view(3, 1, 1)
        mask = torch.all(img == color_tensor, dim=0)
        class_index_map[mask] = class_idx

    return class_index_map


def plot_prediction(x, y, y_pred, color_map, mask_labels):
    fig, axes = plt.subplots(1, 3, figsize=(10, 10))
    items = [(x, 'Image'), (y, 'Truth'), (y_pred, 'Prediction')]

    for i, (img, title) in enumerate(items):
        axes[i].imshow(img[0].cpu().permute(1, 2, 0))
        axes[i].set_title(title)

    plt.tight_layout()
    plt.show()


def compute_accuracy(y, y_pred, device='cpu'):
    # y: (b, h, w)
    # y_pred: (b, c, h, w)
    batch_size = y.shape[0]
    acc_list = []

    for b in range(batch_size):
        y_2d = y[b].to(device)
        y_pred_2d = y_pred[b].to(device)
        y_pred_2d = torch.argmax(y_pred_2d, axis=0)

        correct_pixels = torch.sum(y_2d == y_pred_2d).item()
        total_pixels = y_2d.shape[0] * y_2d.shape[1]

        acc = correct_pixels / total_pixels
        acc_list.append(acc)  

    return np.mean(acc_list)


def get_color_patches(color_map, mask_labels):
    patches = []

    for color, label in color_map.items():
        patches.append(mpatches.Patch(
            color=np.array(color) / 255,
            label=mask_labels[label]
        ))

    return patches
