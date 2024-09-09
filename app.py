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
    unet_checkpoint = torch.load('unet/unet_100ep_checkpoint.pth', weights_only=False)
    unet = UNet(3, 6).to(DEVICE)
    unet.load_state_dict(unet_checkpoint['model_state_dict'])

    res_unet_a_checkpoint = torch.load('res_unet_a/res_unet_a_100ep_checkpoint.pth', weights_only=False)
    res_unet_a = ResUNetA(3, 6).to(DEVICE)
    res_unet_a.load_state_dict(res_unet_a_checkpoint['model_state_dict'])

    return unet, res_unet_a


def gen_example():
    dataset, test_set = load_test_set()
    n = len(test_set)

    img, mask = test_set[random.randrange(0, n)]
    img_col1.subheader("Preview")
    img_col1.image(img.permute(1, 2, 0).numpy())

    pred_img = torch.squeeze(predict(img), 0)
    img_col2.subheader("Prediction")
    img_col2.image(pred_img.cpu().permute(1, 2, 0).numpy())

    patches = get_color_patches(COLOR_MAP, MASK_LABELS)
    fig, ax = plt.subplots(figsize=(8, 0.5))
    ax.set_axis_off()
    plt.legend(handles=patches, loc='center', fontsize=10, ncol=6)
    img_class_placeholder.write(fig)


def predict(x):
    unet, res_unet_a = load_models()
    with torch.no_grad():
        x.to(DEVICE)

        model = unet if model_selection == 'UNet' else res_unet_a
        y_pred = model(torch.unsqueeze(x, 0).to(DEVICE))
        pred_img = one_hot_to_image(y_pred, COLOR_MAP, DEVICE)

    return pred_img


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
    sharpened_img = transforms.functional.adjust_sharpness(saturated_img, 2)
    
    img_tensor = transforms.ToTensor()(sharpened_img)
    
    return sharpened_img, img_tensor


st.header("""
Land Cover Segmentation App
developed by Tom Lam
""")

img_col1, img_col2 = st.columns(2)
img_preview = st.empty()
img_prediction = st.empty()
img_class_placeholder = st.empty()

with st.sidebar:
    model_selection = st.selectbox("Select a model", ("UNet" , "ResUNet-a"))

    st.write("""
    # Use a random unseen image
    """)
    example_button = st.button("Get Segmented Mask", key=0)

    st.write("""
    # Use a custom image
    """)

    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg"])
    custom_image_button = st.button("Get Segmented Mask", key=1)

if example_button:
    gen_example()

if custom_image_button and uploaded_file:
    im = Image.open(uploaded_file).convert("RGB")
    img, img_tensor = preprocessing(im)

    image = np.asarray(img)
    img_preview.image(image, caption="Preview")

    x = img_tensor
    print(x.shape)
    print(x.min(), x.max())
    pred_img = torch.squeeze(predict(x), 0)
    img_prediction.image(pred_img.cpu().permute(1, 2, 0).numpy(), caption="Prediction")