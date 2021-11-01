import os

from PIL import Image
import numpy as np


# This script is used to add black border around the resized image.
# Generated image is used for projector compensation target images


####### Preset for rendered2 setup #######
# offset of top_left corner
def preset_rendered2():
    return 75, 35, 170, 170

####### Preset for rendered3 setup #######
# offset of top_left corner
def preset_rendered3():
    return 72, 30, 170, 170

####### Preset for rendered4 setup #######
# offset of top_left corner
def preset_rendered4():
    return 72,30,170,170

####### Preset for rendered5 setup #######
# offset of top_left corner
def preset_rendered5():
    return 52, 17, 180, 180

####### Preset for compennet++ scene1 setup #######
def preset_compennet_pp_scene1():
    return 112, 56, 77, 77


if __name__ == "__main__":

    result_width = 640
    result_height = 480
    
    # Choose one of the setup functions defined above
    offset_x, offset_y, transformed_width, transformed_height = 206,136,458-206,396-136

    file_list = ["output_ref.png", "output_texture.png"]

    for filename in file_list:
        
        img = Image.open(filename)

        target_arr = np.zeros((result_height, result_width, 3))

        transformed_img = img.resize((transformed_width, transformed_height))
        img_arr = np.asarray(transformed_img)

        for i in range(transformed_width):
            for j in range(transformed_height):
                target_arr[j + offset_y][i + offset_x] = img_arr[j][i]

        target_img = Image.fromarray(np.uint8(target_arr))
        target_img.save("warped_"+filename, "PNG")



