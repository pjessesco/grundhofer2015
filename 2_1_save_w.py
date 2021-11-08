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
color_channel = 3
STEP = 4

print("Reading images...")
ref = read_pngs(REF_PATH)
captured = read_pngs(REF_CAPTURED_WARPED_PATH)
print("Done")

delta = cp.zeros((N, N, STEP, img_width))
for i in range(N):
    for j in range(N):
        if i==j:
            delta[i, j] = cp.ones((STEP, img_width))
        else:
            delta[i,j] = cp.zeros((STEP, img_width))

lambda_val = 0.05

Q_ = np.transpose(captured, (1,2,0,3))
Q_ = np.insert(Q_, 0, 1, axis=3)
# (125, 4, 100, 100)
Q_ = np.transpose(Q_, (2, 3, 0, 1))

O = np.zeros((4, 4, STEP, img_width))

print("Done")

Q_ = cp.asarray(Q_)
O = cp.asarray(O)
ref = cp.asarray(ref)
captured = cp.asarray(captured)

W = cp.zeros((N+4, 3, img_width, img_height))

def process(w_start, w_end):
    w_partial = w_end - w_start

    euclid_dist_mat = cp.zeros((N, N, w_partial, img_height))
    
    for i in range(N):
        for j in range(i, N):
            euclid_dist_mat[i, j] = euclid_dist_mat[j, i] = cp.linalg.norm(captured[i, w_start:w_end] - captured[j, w_start:w_end], axis=2)
    
    rbf_mat = tps_rbf(euclid_dist_mat)

    Q = Q_[:, :, w_start:w_end, :]
    alpha = cp.sum(euclid_dist_mat, axis=(0, 1))
    alpha /= N * N

    K = rbf_mat + (delta[:,:,:w_partial] * lambda_val * alpha * alpha)
    L = cp.concatenate([cp.concatenate([K, Q], axis=1), cp.concatenate([cp.transpose(Q, (1, 0, 2, 3)), O[:,:,:w_partial]], axis=1)], axis=0)

    PO = cp.transpose(cp.concatenate([cp.transpose(ref[:, w_start:w_end, :, :], (0,3,1,2)), cp.zeros((4, 3, w_partial, img_height))], axis=0), (2,3,0,1))
    
    return cp.transpose(cp.linalg.inv(cp.transpose(L, (2,3,0,1))) @ PO, (2,3,0,1))


if __name__ == '__main__':


    assert (ref.shape == captured.shape)
    assert (ref.shape[0] == N)

    params = range(0, img_width, STEP)

    for w in tqdm(params):
        min_w = w
        max_w = min(w+STEP, img_width)
        W[:, :, min_w:max_w, :] = process(min_w, max_w)
        
    cp.save("W", W)

