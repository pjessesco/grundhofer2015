'''
CompenNet++ CNN model
'''

import cv2
import numpy as np
import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_tps
import tqdm
from PIL import Image
import time

import os
import shutil
from common import *


if __name__ == "__main__":

    myapp = myImageDisplayApp()
    l515 = RealSense()

    l515.set_dark()
    time.sleep(10)

    # Warping train images
    offset_x, offset_y, transformed_width, transformed_height = calc_offset(left, up, right, down)

    result_width = 640
    result_height = 480

    INPUT_IMG_PATH = "./WarpingNetOnly_l1+ssim_500_48_1500/"
    if not os.path.isdir(INPUT_IMG_PATH):
        os.mkdir(INPUT_IMG_PATH)

    projected_img_dir = "tps_input_before_cropped/"
    input_path = os.path.join(INPUT_IMG_PATH)
    output_path = os.path.join(projected_img_dir)

    if not os.path.exists(input_path):
        os.makedirs(input_path)

    if not os.path.exists(output_path):            
        os.makedirs(output_path)

    images = os.listdir(input_path)
    images.sort()

    for image in images:
        myapp.emit_image_update(os.path.join(input_path, image))
        l515.get_rgb_image(output_path, image)


    # Crop projected image
    cropped_img_dir = "tps_input/"
    if not os.path.exists(cropped_img_dir):
        os.makedirs(cropped_img_dir)
    
    for image in tqdm.tqdm(os.listdir(projected_img_dir)):
        img = Image.open(projected_img_dir+image)
        img = img.crop((left, up, right, down))
        img.save(cropped_img_dir+image)

    shutil.rmtree("texture_warped")
    shutil.rmtree("proj_input_texture")
    shutil.rmtree("tps_input_before_cropped")
    


