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

import os

import common

# WarpingNet
class WarpingNet(nn.Module):
    def __init__(self, out_size, grid_shape=(6, 6), with_refine=True):
        super(WarpingNet, self).__init__()
        self.grid_shape = grid_shape
        self.out_size = out_size
        self.with_refine = with_refine  # becomes WarpingNet w/o refine if set to false
        self.name = 'WarpingNet' if not with_refine else 'WarpingNet_without_refine'

        # relu
        self.relu = nn.ReLU()
        self.leakyRelu = nn.LeakyReLU(0.1)

        # final refined grid
        self.register_buffer('fine_grid', None)

        # affine params
        self.affine_mat = nn.Parameter(torch.Tensor([1, 0, 0, 0, 1, 0]).view(-1, 2, 3))

        # tps params
        self.nctrl = self.grid_shape[0] * self.grid_shape[1]
        self.nparam = (self.nctrl + 2)
        ctrl_pts = pytorch_tps.uniform_grid(grid_shape)
        self.register_buffer('ctrl_pts', ctrl_pts.view(-1, 2))
        self.theta = nn.Parameter(torch.ones((1, self.nparam * 2), dtype=torch.float32).view(-1, self.nparam, 2) * 1e-3)

        # initialization function, first checks the module type,
        def init_normal(m):
            if type(m) == nn.Conv2d:
                nn.init.normal_(m.weight, 0, 1e-4)

        # grid refinement net
        if self.with_refine:
            self.grid_refine_net = nn.Sequential(
                nn.Conv2d(2, 32, 3, 2, 1),
                self.relu,
                nn.Conv2d(32, 64, 3, 2, 1),
                self.relu,
                nn.ConvTranspose2d(64, 32, 2, 2, 0),
                self.relu,
                nn.ConvTranspose2d(32, 2, 2, 2, 0),
                self.leakyRelu
            )
            self.grid_refine_net.apply(init_normal)
        else:
            self.grid_refine_net = None  # WarpingNet w/o refine

    # initialize WarpingNet's affine matrix to the input affine_vec
    def set_affine(self, affine_vec):
        self.affine_mat.data = torch.Tensor(affine_vec).view(-1, 2, 3)

    # simplify trained model to a single sampling grid for faster testing
    def simplify(self, x):
        # generate coarse affine and TPS grids
        coarse_affine_grid = F.affine_grid(self.affine_mat, torch.Size([1, x.shape[1], x.shape[2], x.shape[3]])).permute((0, 3, 1, 2))
        coarse_tps_grid = pytorch_tps.tps_grid(self.theta, self.ctrl_pts, (1, x.size()[1]) + self.out_size)

        # use TPS grid to sample affine grid
        tps_grid = F.grid_sample(coarse_affine_grid, coarse_tps_grid)

        # refine TPS grid using grid refinement net and save it to self.fine_grid
        if self.with_refine:
            self.fine_grid = torch.clamp(self.grid_refine_net(tps_grid) + tps_grid, min=-1, max=1).permute((0, 2, 3, 1))
        else:
            self.fine_grid = torch.clamp(tps_grid, min=-1, max=1).permute((0, 2, 3, 1))

    def forward(self, x):

        if self.fine_grid is None:
            # not simplified (training/validation)
            # generate coarse affine and TPS grids
            coarse_affine_grid = F.affine_grid(self.affine_mat, torch.Size([1, x.shape[1], x.shape[2], x.shape[3]])).permute((0, 3, 1, 2))
            coarse_tps_grid = pytorch_tps.tps_grid(self.theta, self.ctrl_pts, (1, x.size()[1]) + self.out_size)

            # use TPS grid to sample affine grid
            tps_grid = F.grid_sample(coarse_affine_grid, coarse_tps_grid).repeat(x.shape[0], 1, 1, 1)

            # refine TPS grid using grid refinement net and save it to self.fine_grid
            if self.with_refine:
                fine_grid = torch.clamp(self.grid_refine_net(tps_grid) + tps_grid, min=-1, max=1).permute((0, 2, 3, 1))
            else:
                fine_grid = torch.clamp(tps_grid, min=-1, max=1).permute((0, 2, 3, 1))
        else:
            # simplified (testing)
            fine_grid = self.fine_grid.repeat(x.shape[0], 1, 1, 1)

        # warp
        x = F.grid_sample(x, fine_grid)
        return x


def read_png(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype('float32')
    if img is None:
        print("cannot read ", path)
        exit(-1)
    return torch.tensor([np.transpose((cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0),(2,0,1))])



if __name__ == "__main__":


    offset_x, offset_y, transformed_width, transformed_height = 222,111,200,200
    # TODO : Add border before apply warping

    warping_net = WarpingNet(out_size=(600, 600), with_refine='w/o_refine' not in "CompenNet++")
    warping_net.load_state_dict(torch.load(common.WARPINGNET_PARAM_PATH))
    warping_net.eval()

    print("Warpingnet loaded")


    img = read_png("warped_output_texture.png")
    result = np.transpose(warping_net(img)[0].detach().numpy(), (1, 2, 0))
    cv2.imwrite("./"+os.path.basename("tps_texture_result.png"), cv2.cvtColor(result, cv2.COLOR_BGR2RGB) * 255)

    img = read_png("warped_output_ref.png")
    result = np.transpose(warping_net(img)[0].detach().numpy(), (1, 2, 0))
    cv2.imwrite("./"+os.path.basename("tps_ref_result.png"), cv2.cvtColor(result, cv2.COLOR_BGR2RGB) * 255)



