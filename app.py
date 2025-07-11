import os
import random

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import streamlit as st
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from dataset import SatelliteImageDataset
from res_unet_a import ResUNetA
from unet import UNet
from adunet.adunet import ADUNet
from utils import one_hot_to_image, get_color_patches


COLOR_MAP = {
    (60, 16, 152): 0, # building
    (132, 41, 246): 1, # land
    (110, 193, 228): 2, # road
    (254, 221, 58): 3, # vegetation
    (226, 169, 41): 4, # water
    (155, 155, 155): 5 # unlabeled / unknown
}

MASK_LABELS = ['building', 'land', 'road', 'vegetation', 'water', 'unknown']

# Create reverse mapping: class_id → name
COLOR_ID_TO_NAME = {
    0: 'Building',
    1: 'Land',
    2: 'Road',
    3: 'Vegetation',
    4: 'Water',
    5: 'Unknown'
}

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


@st.cache_resource
def load_test_set():
    dataset = SatelliteImageDataset(
        image_dir='data/images',
        mask_dir='data/masks'
    )

    gen = torch.Generator()
    gen.manual_seed(0)

    _, test_set = random_split(dataset, [0.8, 0.2], generator=gen)

    return dataset, test_set


@st.cache_resource
def load_models():
    unet_checkpoint = torch.load('unet/unet_100ep_checkpoint.pth', map_location=DEVICE, weights_only=False)
    unet = UNet(3, 6).to(DEVICE)
    unet.load_state_dict(unet_checkpoint['model_state_dict'])

    res_unet_a_checkpoint = torch.load('res_unet_a/res_unet_a_100ep_checkpoint.pth', map_location=DEVICE, weights_only=False)
    res_unet_a = ResUNetA(3, 6).to(DEVICE)
    res_unet_a.load_state_dict(res_unet_a_checkpoint['model_state_dict'])

    adunet_checkpoint = torch.load('adunet/adunet_100ep_checkpoint.pth', map_location=DEVICE)
    adunet = ADUNet(3, 6).to(DEVICE)
    adunet.load_state_dict(adunet_checkpoint['model_state_dict'])

    return unet, res_unet_a, adunet


def gen_example():
    dataset, test_set = load_test_set()
    n = len(test_set)

    img, mask = test_set[random.randrange(0, n)]
    img_col1.subheader("Preview")
    img_col1.image(img.permute(1, 2, 0).numpy())

    # pred_img = torch.squeeze(predict(img), 0)
    # img_col2.subheader("Prediction")
    # img_col2.image(pred_img.cpu().permute(1, 2, 0).numpy())

    # # Show % breakdown
    # show_class_percentage(torch.argmax(pred_img, dim=0))
    
    pred_img, class_map = predict(img)
    img_col2.subheader("Prediction")
    img_col2.image(pred_img.squeeze(0).cpu().permute(1, 2, 0).numpy())

    # Accurate % from class_map
    show_class_percentage(class_map)

    patches = get_color_patches(COLOR_MAP, MASK_LABELS)
    fig, ax = plt.subplots(figsize=(8, 0.5))
    ax.set_axis_off()
    plt.legend(handles=patches, loc='center', fontsize=10, ncol=6)
    img_class_placeholder.write(fig)

    
def predict(x):
    unet, res_unet_a, adunet = load_models()
    
    with torch.no_grad():
        x = x.to(DEVICE)

        if model_selection == 'UNet':
            model = unet
        elif model_selection == 'ResUNet-a':
            model = res_unet_a
        else:
            model = adunet

        y_pred = model(torch.unsqueeze(x, 0))  # shape: [1, C, H, W]
        class_map = torch.argmax(y_pred.squeeze(), dim=0)  # shape: [H, W]
        pred_img = one_hot_to_image(y_pred, COLOR_MAP, DEVICE)  # RGB image

    return pred_img, class_map


def show_class_percentage(pred_mask_tensor):
    pred_mask_np = pred_mask_tensor.cpu().numpy()
    unique_classes, counts = np.unique(pred_mask_np, return_counts=True)
    total_pixels = pred_mask_np.size

    st.subheader("🧮 Class-wise Area Breakdown:")
    for cls, cnt in zip(unique_classes, counts):
        percent = (cnt / total_pixels) * 100
        class_name = COLOR_ID_TO_NAME.get(cls, f"Class {cls}")
        st.write(f"🔹 {class_name}: {percent:.2f}%")
    
    # Optional Pie Chart
    fig, ax = plt.subplots()
    ax.pie(counts, labels=[COLOR_ID_TO_NAME[c] for c in unique_classes], autopct='%1.1f%%')
    ax.axis('equal')
    st.pyplot(fig)


def preprocessing(im):
    width, height = im.size
    target = min(width, height)
    
    left = (width - target) / 2
    top = (height - target) / 2
    right = (width + target) / 2
    bottom = (height + target) / 2

    croped_img = im.crop((left, top, right, bottom))
    resized_img = transforms.Resize((512, 512))(croped_img)
    saturated_img = transforms.ColorJitter(contrast=(1.25, 1.25))(resized_img)
    sharpened_img = transforms.functional.adjust_sharpness(saturated_img, 3)
    
    img_tensor = transforms.ToTensor()(sharpened_img)
    
    return sharpened_img, img_tensor


st.header("""
Land Cover Segmentation App
""")

img_col1, img_col2 = st.columns(2)
img_class_placeholder = st.empty()

with st.sidebar:
    model_selection = st.selectbox("Select a model", ("UNet", "ResUNet-a", "ADUNet"))


    st.write("""
    # Use a random unseen image
    The unseen image is selected from the test dataset, where it shares a similar color distribution with the partition of the dataset used for training.
    """)
    example_button = st.button("Get Segmented Mask", key=0)

    st.write("""
    # Use a custom image
    For demostration purpose only, the model does not generalize well with real-world data. The model works best with images that consist of multiple land cover types. 
    """)

    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg"])
    custom_image_button = st.button("Get Segmented Mask", key=1)

if example_button:
    gen_example()

if custom_image_button and uploaded_file:
    im = Image.open(uploaded_file).convert("RGB")
    img, img_tensor = preprocessing(im)

    image = np.asarray(img)
    img_col1.subheader("Preview")
    img_col1.image(image)

    x = img_tensor
    pred_img, class_map = predict(x)
    pred_img = torch.squeeze(pred_img, 0)  # only on the image
    img_col2.subheader("Prediction")
    img_col2.image(pred_img.cpu().permute(1, 2, 0).numpy())

    # Show % breakdown
    show_class_percentage(class_map)

    patches = get_color_patches(COLOR_MAP, MASK_LABELS)
    fig, ax = plt.subplots(figsize=(8, 0.5))
    ax.set_axis_off()
    plt.legend(handles=patches, loc='center', fontsize=10, ncol=6)
    img_class_placeholder.write(fig)