import glob
import numpy as np
import cupy as cp
from scipy import linalg
import cv2
import scipy
from tqdm import tqdm

img_width = 600
img_height = 600

def tps_rbf(d):
    return cp.where(d==0, 0, d*d*cp.log2(d))

def read_png(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype('float32')
    img = cv2.resize(img, (img_width, img_height))
    if img is None:
        print("cannot read ", path)
        exit(-1)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0

def read_pngs(dir):

    filelist = glob.glob(dir+"*.png")
    filelist.sort()

    imgs = []

    for filename in filelist:
        img = read_png(filename)
        imgs.append(img)

    return np.array(imgs)


N = 500

REF_PATH = "./texture/"
REF_CAPTURED_WARPED_PATH = "./tps_input/"
INPUT_IMG_PATH = "./ref.png"
color_channel = 3
STEP = 4

print("Reading images...")
captured = read_pngs(REF_CAPTURED_WARPED_PATH)
input_img = read_png(INPUT_IMG_PATH)
print("Done")

output_img = np.zeros(input_img.shape).astype('float32')

O = np.zeros((4, 4, STEP, img_width))

W = cp.load("W.npy")
print(W.shape)

print("Preprocess common terms")
euclid_dist_input_captured_mat = np.zeros((N, img_width, img_height))
for i in range(N):
    euclid_dist_input_captured_mat[i] = np.linalg.norm(input_img - captured[i], axis=2)

print("Done")

euclid_dist_input_captured_mat = cp.asarray(euclid_dist_input_captured_mat)
input_img = cp.asarray(input_img)

def process():

    ret_val = cp.sum(cp.reshape(cp.concatenate([tps_rbf(euclid_dist_input_captured_mat), cp.zeros((4, img_width, img_height))], axis=0), (N+4, 1, img_width, img_height)) * W, axis=0).get()

    partial_input = np.reshape(input_img, (1, img_width, img_height, 3))

    ret_val += (W[N]).get()
    ret_val += (W[N + 1] * partial_input[:,:,:,0]).get()
    ret_val += (W[N + 2] * partial_input[:,:,:,1]).get()
    ret_val += (W[N + 3] * partial_input[:,:,:,2]).get()

    return np.transpose(ret_val, (1,2,0))


if __name__ == '__main__':

    params = range(0, img_width, STEP)

    output_img = process().astype('float32')
        
    print(output_img.shape)
    cv2.imwrite("output_texture.png", cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)*255)
