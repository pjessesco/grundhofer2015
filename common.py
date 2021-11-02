import cv2
import numpy as np
import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_tps
import tqdm
from PIL import Image

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import QObject, pyqtSignal

import threading
import socket
import time
import sys
import os

WARPINGNET_PARAM_PATH = "set1_plane_CompenNet++_l1+ssim_500_48_1500_0.001_0.2_1000_0.0001.pth_warpingnet.pth"

data_dir = "data/"
left, up, right, down = 265, 100, 467, 295

def read_png(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype('float32')
    if img is None:
        print("cannot read ", path)
        exit(-1)
    return torch.tensor([np.transpose((cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0),(2,0,1))])

def calc_offset(x1, y1, x2, y2):
    return x1, y1, x2-x1, y2-y1


class iPadLiDARDevice():

    def __init__(self, host, port=12345):        
        self.__clientSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.__clientSocket.connect((host, port))
        self.__tail_utf8 = "__TAIL_TAIL_TAIL__".encode('utf-8')
        
        print("iPadLiDARDevice : Check connectivity")
        self.__request('dummy')
        self.__request('dummy')
        print("iPadLiDARDevice : Done")

    def __request(self, signal):
        print("send signal : ",signal)
        self.__clientSocket.send(signal.encode())
        print("receiving..")

        data = self.__clientSocket.recv(50000)

        while self.__tail_utf8 not in data:
            print("partially received...")
            data += self.__clientSocket.recv(50000)

        print("whole data received : ", len(data))
        # if not sleep app crashes
        time.sleep(1)
        return data

    def get_rgb_image(self, dir, filename):
        img_data = self.__request('rgb')
        path = os.path.join(dir, filename)
        f = open(path, 'wb')
        f.write(img_data)
        f.close()

        rgb = cv2.imread(path, cv2.IMREAD_COLOR)[:, :, ::-1]
        return rgb

# Reference : https://stackoverflow.com/a/59539843
class myImageDisplayApp (QObject):

    # Define the custom signal
    # https://www.riverbankcomputing.com/static/Docs/PyQt5/signals_slots.html#the-pyqtslot-decorator
    signal_update_image = pyqtSignal(str)

    def __init__ (self):

        super().__init__()

        # Setup the seperate thread 
        # https://stackoverflow.com/a/37694109/4988010
        self.thread = threading.Thread(target=self.run_img_widget_in_background) 
        self.thread.daemon = True
        self.thread.start()

    def run_img_widget_in_background(self):
        self.app = QApplication(sys.argv)
        self.my_bg_qt_app = qtAppWidget(main_thread_object=self)
        self.app.exec_()

    def emit_image_update(self, pattern_file=None):
        print('emit_image_update signal')
        self.signal_update_image.emit(pattern_file)

class qtAppWidget (QLabel):

    def __init__ (self, main_thread_object):

        super().__init__()

        # Connect the singal to slot
        main_thread_object.signal_update_image.connect(self.updateImage)

        self.setupGUI()

    def setupGUI(self):

        self.app = QApplication.instance()

        # Get avaliable screens/monitors
        # https://doc.qt.io/qt-5/qscreen.html
        # Get info on selected screen 
        self.selected_screen = 1            # Select the desired monitor/screen

        self.screens_available = self.app.screens()
        self.screen = self.screens_available[self.selected_screen]
        self.screen_width = self.screen.size().width()
        self.screen_height = self.screen.size().height()

        # Create a black image for init 
        self.pixmap = QPixmap(self.screen_width, self.screen_height)
        self.pixmap.fill(QColor('black'))

        # Create QLabel object
        self.img_widget = QLabel()

        # Varioius flags that can be applied to make displayed window frameless, fullscreen, etc...
        # https://doc.qt.io/qt-5/qt.html#WindowType-enum
        # https://doc.qt.io/qt-5/qt.html#WidgetAttribute-enum
        self.img_widget.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowDoesNotAcceptFocus | Qt.WindowStaysOnTopHint)
        
        # Hide mouse cursor 
        self.img_widget.setCursor(Qt.BlankCursor)       

        self.img_widget.setStyleSheet("background-color: black;") 

        self.img_widget.setGeometry(0, 0, self.screen_width, self.screen_height)            # Set the size of Qlabel to size of the screen
        self.img_widget.setWindowTitle('myImageDisplayApp')
        self.img_widget.setAlignment(Qt.AlignCenter | Qt.AlignTop) #https://doc.qt.io/qt-5/qt.html#AlignmentFlag-enum                         
        self.img_widget.setPixmap(self.pixmap)
        self.img_widget.show()

        # Set the screen on which widget is on
        self.img_widget.windowHandle().setScreen(self.screen)
        # Make full screen 
        self.img_widget.showFullScreen()
        

    def updateImage(self, pattern_file=None):
        print('Pattern file given: ', pattern_file)
        self.img_widget.clear()                     # Clear all existing content of the QLabel
        
        self.pixmap.fill(QColor('green'))
        pixmap = QPixmap(pattern_file).scaled(self.screen_width,self.screen_height, Qt.KeepAspectRatio)         # Update pixmap with desired image
        self.pixmap = pixmap

        self.img_widget.setPixmap(self.pixmap)      # Show desired image on Qlabel

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
