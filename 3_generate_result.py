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

    # Warping train images
    offset_x, offset_y, transformed_width, transformed_height = calc_offset(left, up, right, down)

    result_width = 640
    result_height = 480

    img = Image.open("output_texture.png")
    target_arr = np.zeros((result_height, result_width, 3))

    transformed_img = img.resize((transformed_width, transformed_height))
    img_arr = np.asarray(transformed_img)

    for i in range(transformed_width):
        for j in range(transformed_height):
            target_arr[j + offset_y][i + offset_x] = img_arr[j][i]

    target_img = Image.fromarray(np.uint8(target_arr))
    target_img.save("warped_output_texture.png", "PNG")

    # Inference projector input image using warpingnet
    warping_net = WarpingNet(out_size=(600, 600), with_refine='w/o_refine' not in "CompenNet++")
    warping_net.load_state_dict(torch.load(WARPINGNET_PARAM_PATH))
    warping_net.eval()

    print("Warpingnet loaded")

    img = read_png("warped_output_texture.png")
    result = np.transpose(warping_net(img)[0].detach().numpy(), (1,2,0))
    cv2.imwrite("tps_projector_input.png", cv2.cvtColor(result, cv2.COLOR_BGR2RGB) * 255)

    


