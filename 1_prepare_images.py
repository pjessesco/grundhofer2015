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
from common import *


if __name__ == "__main__":

    myapp = myImageDisplayApp()
    iPad = iPadLiDARDevice('192.168.0.17')

    left, up, right, down = 174, 138, 462, 396

    # Warping train images
    offset_x, offset_y, transformed_width, transformed_height = calc_offset(left, up, right, down)

    result_width = 640
    result_height = 480

    src_path = "texture"
    dst_path = "texture_warped"
    if not os.path.isdir(dst_path):
        os.mkdir(dst_path)
    file_list = [f for f in os.listdir(src_path) if os.path.isfile(os.path.join(src_path, f))]
    for filename in tqdm.tqdm(file_list):
        src_file_path = os.path.join(src_path, filename)
        dst_file_path = os.path.join(dst_path, filename)

        img = Image.open(src_file_path)

        target_arr = np.zeros((result_height, result_width, 3))

        transformed_img = img.resize((transformed_width, transformed_height))
        img_arr = np.asarray(transformed_img)

        for i in range(transformed_width):
            for j in range(transformed_height):
                target_arr[j + offset_y][i + offset_x] = img_arr[j][i]

        target_img = Image.fromarray(np.uint8(target_arr))
        target_img.save(dst_file_path, "PNG")


    print()
    print("Copy :")
    print("  CompenNet-plusplus/checkpoint/~~~.pth_warpingnet.pth -> grundhofer2015~~~.pth_warpingnet.pth")
    print()

    # Inference projector input image using warpingnet
    warping_net = WarpingNet(out_size=(600, 600), with_refine='w/o_refine' not in "CompenNet++")
    warping_net.load_state_dict(torch.load(WARPINGNET_PARAM_PATH))
    warping_net.eval()

    INPUT_IMG_PATH = "./texture_warped/"
    OUTPUT_IMG_PATH = "./proj_input_texture/"
    if not os.path.isdir(OUTPUT_IMG_PATH):
        os.mkdir(OUTPUT_IMG_PATH)

    assert(os.path.isdir(INPUT_IMG_PATH))
    assert(os.path.isdir(OUTPUT_IMG_PATH))

    print("Warpingnet loaded")

    filelist = glob.glob(INPUT_IMG_PATH + "*.png")
    filelist.sort()
    for filename in tqdm.tqdm(filelist):
        img = read_png(filename)
        result = np.transpose(warping_net(img)[0].detach().numpy(), (1,2,0))
        cv2.imwrite(OUTPUT_IMG_PATH + os.path.basename(filename), cv2.cvtColor(result, cv2.COLOR_BGR2RGB) * 255)


    # Project warpingnet-inferenced images and capture

    projected_img_dir = "tps_input_before_cropped"
    input_path = os.path.join(OUTPUT_IMG_PATH)
    output_path = os.path.join(projected_img_dir)

    if not os.path.exists(input_path):
        os.makedirs(input_path)

    if not os.path.exists(output_path):            
        os.makedirs(output_path)

    images = os.listdir(input_path)
    images.sort()

    for image in images:
        myapp.emit_image_update(os.path.join(input_path, image))
        time.sleep(0.2)
        iPad.get_rgb_image(output_path, image)


    # Crop projected image
    cropped_img_dir = "tps_input"
    if not os.path.exists(cropped_img_dir):
        os.makedirs(cropped_img_dir)
    
    for image in tqdm.tqdm(os.listdir(cropped_img_dir)):
        img = Image.open(image)
        img = img.crop((left, up, right, down))
        img.save(image)

    


