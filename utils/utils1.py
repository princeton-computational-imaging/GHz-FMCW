import sys
import json
import time
import torch
import numpy as np
import torchvision.utils as vutils
from glob import glob
import logging
import optparse
import os
from os.path import join
import shutil
import matplotlib
import matplotlib.cm
import cv2
import torch.nn as nn
from scipy import ndimage as ndi
from os.path import exists
import csv as csv

speed_of_light = 299792458
lamda = speed_of_light/71500000

def load_bg_depth(root_path):
    filename = root_path+"depth/scene_wall_depth.csv"
    file_exists = exists(filename)

    if file_exists == True:
        with open(filename, newline='') as f:
            reader = csv.reader(f)
            depth = list(reader)
        depth = np.array(depth)
        depth = depth.astype(float)
        depth = depth*10
    else:
        print("file does not exist.")
    return depth

class GradientLoss(nn.Module):
    def __init__(self, phase, depth, obj_mask, smooth, weights = [1,1], scale = 1):
        super(GradientLoss, self).__init__()
        
        phase = phase.to(torch.float32)
        dh_depth, dv_depth = spatial_gradient_tensor(depth)
        dh, dv = spatial_gradient_tensor(phase2depth(phase, 7.15e9))
        
        false_mask = torch.zeros_like(dh)
        true_mask = torch.ones_like(dh)
        
        dh_edges = torch.where(torch.logical_and(dh<5 ,dh>-5), false_mask, true_mask)
        dv_edges = torch.where(torch.logical_and(dv<5 ,dv>-5), false_mask, true_mask)
        dh = torch.where(torch.logical_and(dh_edges == 1, obj_mask == 1), false_mask-1, dh)
        dv = torch.where(torch.logical_and(dv_edges == 1, obj_mask == 1), false_mask-1, dv)
        dh_unwrapped = torch.where(torch.logical_and(dh_edges == 1, obj_mask == 0), dh_depth, dh) 
        dv_unwrapped = torch.where(torch.logical_and(dv_edges == 1, obj_mask == 0), dv_depth, dv) 
        if smooth:
            dh_unwrapped = self.smoothing(dh_unwrapped)
            dv_unwrapped = self.smoothing(dv_unwrapped)
        
        
        self.target_dh, self.target_dv = dh_unwrapped.detach(), dv_unwrapped.detach()
        self.weights = weights
        self.obj_mask = obj_mask 
        self.false_mask = false_mask
        self.true_mask = true_mask
        
    def smoothing(self, d):
        d_med = torch.from_numpy(ndi.median_filter(d.numpy(), size=3))
        return d_med
    
    def forward(self, input):
        dh, dv = spatial_gradient_tensor(input)
        loss = self.weights[0]*F.mse_loss(dh.float(), self.target_dh.float()) + \
               self.weights[1]*F.mse_loss(dv.float(), self.target_dv.float())
        return loss


class WrapEdgeLoss(nn.Module):
    def __init__(self, phase, obj_mask, smooth, weights = [1,1], scale = 1):
        super(WrapEdgeLoss, self).__init__()
        
        phase = phase.to(torch.float32)
        dh, dv = spatial_gradient_tensor(phase2depth(phase, 7.15e9))
        
        false_mask = torch.zeros_like(dh)
        true_mask = torch.ones_like(dh)
        
        if smooth:
            dh_avg = self.smoothing(dh)
            dv_avg = self.smoothing(dv)    
        else:
            dh_avg = dh #self.smoothing(dh)
            dv_avg = dv #self.smoothing(dv)
        dh_bg = torch.where(obj_mask == 1, dh_avg, false_mask)
        dv_bg = torch.where(obj_mask == 1, dv_avg, false_mask)
        dh = torch.where(obj_mask == 1, dh_bg, dh)
        dv = torch.where(obj_mask == 1, dv_bg, dv)
        
        self.target_dh, self.target_dv = dh.detach(), dv.detach()
        self.weights = weights
        self.obj_mask = obj_mask
        self.target_dh_masked = torch.where(torch.logical_and(dh > -5, dh < 15), false_mask, dh)
        self.target_dv_masked = torch.where(torch.logical_and(dv > -5, dv < 15), false_mask, dv)
        
        
    def smoothing(self, d):
        d_med = torch.from_numpy(ndi.median_filter(d.numpy(), size=3))
        return d_med
    
    def forward(self, input):
        dh, dv = spatial_gradient_tensor(input)
        false_mask = torch.zeros_like(dh)
        
        target_dh_masked = torch.where(self.target_dh_masked == 0, \
                                       false_mask, torch.mul(torch.sign(dh), torch.abs(self.target_dh_masked)))
        target_dv_masked = torch.where(self.target_dv_masked == 0, \
                                       false_mask, torch.mul(torch.sign(dv), torch.abs(self.target_dv_masked)))
        
        loss = self.weights[0]*F.mse_loss(dh, target_dh_masked) + \
               self.weights[1]*F.mse_loss(dv, target_dv_masked)
        
        return loss

def bg_finetune(root_path, depth, mask):
    bg_depth = load_bg_depth(root_path)
    final_depth = np.where(mask == 1, bg_depth, depth)
    return final_depth

class GradientLossFinetune(nn.Module):
    def __init__(self, phase, depth, obj_mask, weights = [1,1], scale = 1):
        super(GradientLossFinetune, self).__init__()
        
        phase = phase.to(torch.float32)
        dh_depth, dv_depth = spatial_gradient_tensor(depth)
        dh, dv = spatial_gradient_tensor(phase2depth(phase, 7.15e9))
        
        false_mask = torch.zeros_like(dh)
        true_mask = torch.ones_like(dh)
        
        dh_edges = torch.where(torch.logical_and(dh<8 ,dh>-5), false_mask, true_mask)
        dv_edges = torch.where(torch.logical_and(dv<10 ,dv>-10), false_mask, true_mask)
        
        dh = torch.where(torch.logical_and(dh_edges == 1, obj_mask == 1), false_mask-1, dh) ###dh<8 ,dh>-5
        dv = torch.where(torch.logical_and(dv_edges == 1, obj_mask == 1), false_mask-1, dv) ###
        dh_unwrapped = torch.where(torch.logical_and(dh_edges == 1, obj_mask == 0), dh_depth, dh) 
        dv_unwrapped = torch.where(torch.logical_and(dv_edges == 1, obj_mask == 0), dv_depth, dv) 
        
        
        self.target_dh, self.target_dv = dh_unwrapped.detach(), dv_unwrapped.detach()
        self.weights = weights
        self.obj_mask = obj_mask        
        self.false_mask = false_mask
        self.true_mask = true_mask
        
    def smoothing(self, d):
        d_med = torch.from_numpy(ndi.median_filter(d.numpy(), size=3))
        return d_med
    
    def forward(self, input):
        dh, dv = spatial_gradient_tensor(input)
        return self.weights[0]*F.mse_loss(dh.float(), self.target_dh.float()) + \
               self.weights[1]*F.mse_loss(dv.float(), self.target_dv.float())




def plot_normal(dv, dh):
    gy, gx = dv, dh #100x100, 100x100  
    normal = np.dstack((-gy, -gx, np.ones_like(gx)))
    n = np.linalg.norm(normal, axis=2)
    normal[:, :, 0] = normal[:, :, 0]/n
    normal[:, :, 1] = normal[:, :, 1]/n
    normal[:, :, 2] = normal[:, :, 2]/n
    return normal

def phase2depth(phase, freq): # convert
    """Convert phase map to depth map.
    Args:
        phase (tensor [B,1,H,W]): Phase map (radian)
        freq ([type]): Frequency (Hz)
    Returns:
        depth (tensor [B,1,H,W]): Depth map (mm)
    """
    depth = (1000*speed_of_light*phase)/(4*np.pi*freq)
    return depth

def spatial_gradient(x):
    dv = np.zeros_like(x)
    dv[:-1, :] = x[1:, :] - x[:-1, :]
    dh = np.zeros_like(x)
    dh[:, :-1] = x[:, 1:] - x[:, :-1]
    
    return [dh, dv]

def spatial_gradient_tensor(x):
    dv = torch.zeros_like(x)
    dv[:-1, :] = x[1:, :] - x[:-1, :]
    dh = torch.zeros_like(x)
    dh[:, :-1] = x[:, 1:] - x[:, :-1]
    
    return [dh, dv]
def colorize(value, vmin=None, vmax=None, cmap=None):
    """
    A utility function for Torch/Numpy that maps a grayscale image to a matplotlib
    colormap for use with TensorBoard image summaries.
    By default it will normalize the input value to the range 0..1 before mapping
    to a grayscale colormap.
    Arguments:
      - value: 2D Tensor of shape [height, width] or 3D Tensor of shape
        [height, width, 1].
      - vmin: the minimum value of the range used for normalization.
        (Default: value minimum)
      - vmax: the maximum value of the range used for normalization.
        (Default: value maximum)
      - cmap: a valid cmap named for use with matplotlib's `get_cmap`.
        (Default: Matplotlib default colormap)
    
    Returns a 4D uint8 tensor of shape [height, width, 4].
    """
    # normalize
    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax
    if vmin!=vmax:
        value = (value - vmin) / (vmax - vmin) # vmin..vmax
    else:
        # Avoid 0-division
        value = value*0.
        
    cmapper = matplotlib.cm.get_cmap(cmap)
    value = cmapper(value.cpu().detach().numpy()) # (nxmx4)
    value = value[...,:3].swapaxes(1,4).squeeze(4) # remove alpha
    return value

def tensor_to_pcd(arr):
    B, C, H, W = arr.shape
    arr = arr.squeeze().reshape(B,H*W,1)
    x = torch.arange(0,W, device=arr.device)
    x = x[None,:,None].repeat(B,H,1)
    y = torch.arange(0,H, device=arr.device)
    y = y[None,:,None].repeat(B,1,1).repeat_interleave(W, dim=1)
    pcd = torch.cat([x,y,arr], dim=2)
    return pcd

def read_text_lines(filepath):
    with open(filepath, "r") as f:
        lines = f.readlines()
    lines = [l.rstrip() for l in lines]
    return lines


def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)  # explicitly set exist_ok when multi-processing

def save_args(args, filename="args.json"):
    args_dict = vars(args)
    check_path(args.checkpoint_dir)
    save_path = join(args.checkpoint_dir, filename)

    with open(save_path, "w") as f:
        json.dump(args_dict, f, indent=4, sort_keys=False)
        
def save_net_files(args):
    dest = join(args.checkpoint_dir, "net_files/")
    check_path(dest)
    shutil.copy(join("net", "ToFSimulator.py"), dest)
    shutil.copy(join("net", "PhaseToPhaseNet.py"), dest)
    shutil.copy(join("net", "arch.py"), dest)
        
def count_parameters(net):
    num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return num

def save_checkpoint(save_path, optimizer, net, epoch, num_iter,
                    loss, filename=None, save_optimizer=True):
    # Network
    net_state = {
        "epoch": epoch,
        "num_iter": num_iter,
        "loss": loss,
        "state_dict": net.state_dict()
    }
    net_filename = "net_epoch_{:0>3d}.pt".format(epoch) if filename is None else filename
    net_save_path = join(save_path, net_filename)
    torch.save(net_state, net_save_path)
    
    # Optimizer
    if save_optimizer:
        optimizer_state = {
            "epoch": epoch,
            "num_iter": num_iter,
            "loss": loss,
            "state_dict": optimizer.state_dict()
        }
        optimizer_name = net_filename.replace("net", "optimizer")
        optimizer_save_path = join(save_path, optimizer_name)
        torch.save(optimizer_state, optimizer_save_path)


def load_checkpoint(net, pretrained_path, return_epoch_iter=False, resume=False, no_strict=False):
    if pretrained_path is not None:
        if torch.cuda.is_available():
            state = torch.load(pretrained_path, map_location="cuda")
        else:
            state = torch.load(pretrained_path, map_location="cpu")

        net.load_state_dict(state["state_dict"])  # optimizer has no argument `strict`

        if return_epoch_iter:
            epoch = state["epoch"] if "epoch" in state.keys() else None
            num_iter = state["num_iter"] if "num_iter" in state.keys() else None
            return epoch, num_iter


def resume_latest_ckpt(checkpoint_dir, net, net_name):
    ckpts = sorted(glob(checkpoint_dir + "/" + net_name + "*.pt"))

    if len(ckpts) == 0:
        raise RuntimeError("=> No checkpoint found while resuming training")

    latest_ckpt = ckpts[-1]
    print("=> Resume latest {0} checkpoint: {1}".format(net_name, os.path.basename(latest_ckpt)))
    epoch, num_iter = load_checkpoint(net, latest_ckpt, True, True)

    return epoch, num_iter

def save_images(logger, mode_tag, images_dict, global_step):
    for tag, values in images_dict.items():
        if not isinstance(values, list) and not isinstance(values, tuple):
            values = [values]
        for idx, value in enumerate(values):
            if len(value.shape) == 3:
                value = value[:, np.newaxis, :, :]
            value = value[:1]
            value = torch.from_numpy(value) 

            image_name = "{}/{}".format(mode_tag, tag)
            if len(values) > 1:
                image_name = image_name + "_" + str(idx)
            logger.add_image(image_name, vutils.make_grid(value, padding=0, nrow=1, normalize=False, scale_each=False),
                             global_step)
            
def tensor2numpy(var_dict):
    for key, vars in var_dict.items():
        if isinstance(vars, np.ndarray):
            var_dict[key] = vars
        elif isinstance(vars, torch.Tensor):
            var_dict[key] = vars.data.cpu().numpy()
        else:
            raise NotImplementedError("invalid input type for tensor2numpy")

    return var_dict

def get_all_data_folders(base_dir=None):
    if base_dir is None:
        base_dir = os.getcwd()

    data_folders = []
    categories = [d for d in os.listdir(base_dir) if os.path.isdir(join(base_dir, d))]
    for category in categories:
        for scene in os.listdir(join(base_dir, category)):
            data_folder = join(*[base_dir, category, scene])
            if os.path.isdir(data_folder):
                data_folders.append(data_folder)

    return data_folders


def get_comma_separated_args(option, opt, value, parser):
    values = [v.strip() for v in value.split(",")]
    setattr(parser.values, option.dest, values)


def parse_options():
    parser = optparse.OptionParser()
    parser.add_option("-d", "--date_folder", type="string", action="callback", callback=get_comma_separated_args,
                      dest="data_folders", help="e.g. stratified/dots,test/bedroom")
    options, remainder = parser.parse_args()

    if options.data_folders is None:
        options.data_folders = get_all_data_folders(os.getcwd())
    else:
        options.data_folders = [os.path.abspath("%s") % d for d in options.data_folders]
        for f in options.data_folders:
            print(f)

    return options.data_folders



def gkern(l=5, sig=1.):
    """\
    creates gaussian kernel with side length l and a sigma of sig
    """

    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sig))
    return kernel




################################# CANNY EDGE FILTER #################################


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2

# https://towardsdatascience.com/implement-canny-edge-detection-from-scratch-with-pytorch-a1cccfa58bed

def get_gaussian_kernel(k=3, mu=0, sigma=1, normalize=True):
    # compute 1 dimension gaussian
    gaussian_1D = np.linspace(-1, 1, k)
    # compute a grid distance from center
    x, y = np.meshgrid(gaussian_1D, gaussian_1D)
    distance = (x ** 2 + y ** 2) ** 0.5

    # compute the 2 dimension gaussian
    gaussian_2D = np.exp(-(distance - mu) ** 2 / (2 * sigma ** 2))
    gaussian_2D = gaussian_2D / (2 * np.pi *sigma **2)

    # normalize part (mathematically)
    if normalize:
        gaussian_2D = gaussian_2D / np.sum(gaussian_2D)
    return gaussian_2D

def get_sobel_kernel(k=3):
    # get range
    range = np.linspace(-(k // 2), k // 2, k)
    # compute a grid the numerator and the axis-distances
    x, y = np.meshgrid(range, range)
    sobel_2D_numerator = x
    sobel_2D_denominator = (x ** 2 + y ** 2)
    sobel_2D_denominator[:, k // 2] = 1  # avoid division by zero
    sobel_2D = sobel_2D_numerator / sobel_2D_denominator
    return sobel_2D

def get_thin_kernels(start=0, end=360, step=45):
        k_thin = 3  # actual size of the directional kernel
        # increase for a while to avoid interpolation when rotating
        k_increased = k_thin + 2

        # get 0° angle directional kernel
        thin_kernel_0 = np.zeros((k_increased, k_increased))
        thin_kernel_0[k_increased // 2, k_increased // 2] = 1
        thin_kernel_0[k_increased // 2, k_increased // 2 + 1:] = -1

        # rotate the 0° angle directional kernel to get the other ones
        thin_kernels = []
        for angle in range(start, end, step):
            (h, w) = thin_kernel_0.shape
            # get the center to not rotate around the (0, 0) coord point
            center = (w // 2, h // 2)
            # apply rotation
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)
            kernel_angle_increased = cv2.warpAffine(thin_kernel_0, rotation_matrix, (w, h), cv2.INTER_NEAREST)

            # get the k=3 kerne
            kernel_angle = kernel_angle_increased[1:-1, 1:-1]
            is_diag = (abs(kernel_angle) == 1)      # because of the interpolation
            kernel_angle = kernel_angle * is_diag   # because of the interpolation
            thin_kernels.append(kernel_angle)
        return thin_kernels

class CannyFilter(nn.Module):
    def __init__(self,
                 k_gaussian=3,
                 mu=0,
                 sigma=1,
                 k_sobel=3,
                 device='cpu'):
        super(CannyFilter, self).__init__()
        # device
        self.device = device

        # gaussian

        gaussian_2D = get_gaussian_kernel(k_gaussian, mu, sigma)
        self.gaussian_filter = nn.Conv2d(in_channels=1,
                                         out_channels=1,
                                         kernel_size=k_gaussian,
                                         padding=k_gaussian // 2,
                                         bias=False)
        self.gaussian_filter.weight[:] = torch.from_numpy(gaussian_2D)
        self.gaussian_filter = self.gaussian_filter.to(self.device)
        # sobel

        sobel_2D = get_sobel_kernel(k_sobel)
        self.sobel_filter_x = nn.Conv2d(in_channels=1,
                                        out_channels=1,
                                        kernel_size=k_sobel,
                                        padding=k_sobel // 2,
                                        bias=False)
        self.sobel_filter_x.weight[:] = torch.from_numpy(sobel_2D)
        self.sobel_filter_x = self.sobel_filter_x.to(self.device)

        self.sobel_filter_y = nn.Conv2d(in_channels=1,
                                        out_channels=1,
                                        kernel_size=k_sobel,
                                        padding=k_sobel // 2,
                                        bias=False)
        self.sobel_filter_y.weight[:] = torch.from_numpy(sobel_2D.T)
        self.sobel_filter_y = self.sobel_filter_y.to(self.device)


        # thin

        thin_kernels = get_thin_kernels()
        directional_kernels = np.stack(thin_kernels)

        self.directional_filter = nn.Conv2d(in_channels=1,
                                            out_channels=8,
                                            kernel_size=thin_kernels[0].shape,
                                            padding=thin_kernels[0].shape[-1] // 2,
                                            bias=False)
        self.directional_filter.weight[:, 0] = torch.from_numpy(directional_kernels)
        self.directional_filter = self.directional_filter.to(self.device)
        # hysteresis

        hysteresis = np.ones((3, 3)) + 0.25
        self.hysteresis = nn.Conv2d(in_channels=1,
                                    out_channels=1,
                                    kernel_size=3,
                                    padding=1,
                                    bias=False)
        self.hysteresis.weight[:] = torch.from_numpy(hysteresis)
        self.hysteresis = self.hysteresis.to(self.device)
        
        self.pad_10 = nn.ReflectionPad2d(10)
        self.unpad_10 = nn.ReflectionPad2d(-10)

    def forward(self, img, low_threshold=None, high_threshold=None, hysteresis=False):
        # set the setps tensors
        img = self.pad_10(img)
        B, C, H, W = img.shape
        blurred = torch.zeros((B, C, H, W)).to(self.device)
        grad_x = torch.zeros((B, 1, H, W)).to(self.device)
        grad_y = torch.zeros((B, 1, H, W)).to(self.device)
        grad_magnitude = torch.zeros((B, 1, H, W)).to(self.device)
        grad_orientation = torch.zeros((B, 1, H, W)).to(self.device)

        # gaussian
        for c in range(C):
            blurred[:, c:c+1] = self.gaussian_filter(img[:, c:c+1])
            grad_x = grad_x + self.sobel_filter_x(blurred[:, c:c+1])
            grad_y = grad_y + self.sobel_filter_y(blurred[:, c:c+1])
        grad_x = self.unpad_10(grad_x)
        grad_y = self.unpad_10(grad_y)

        # thick edges

        grad_x, grad_y = grad_x / C, grad_y / C
        grad_magnitude = (grad_x ** 2 + grad_y ** 2) ** 0.5
        grad_orientation = torch.atan(grad_y / grad_x)
        grad_orientation = grad_orientation * (360 / np.pi) + 180 # convert to degree
        grad_orientation = torch.round(grad_orientation / 45) * 45  # keep a split by 45

        # thin edges

        directional = self.directional_filter(grad_magnitude)
        # get indices of positive and negative directions
        positive_idx = (grad_orientation / 45) % 8
        negative_idx = ((grad_orientation / 45) + 4) % 8
        thin_edges = grad_magnitude.clone()
        # non maximum suppression direction by direction
        for pos_i in range(4):
            neg_i = pos_i + 4
            # get the oriented grad for the angle
            is_oriented_i = (positive_idx == pos_i) * 1
            is_oriented_i = is_oriented_i + (positive_idx == neg_i) * 1
            pos_directional = directional[:, pos_i]
            neg_directional = directional[:, neg_i]
            selected_direction = torch.stack([pos_directional, neg_directional])

            # get the local maximum pixels for the angle
            is_max = selected_direction.min(dim=0)[0] > 0.0
            is_max = torch.unsqueeze(is_max, dim=1)

            # apply non maximum suppression
            to_remove = (is_max == 0) * 1 * (is_oriented_i) > 0
            thin_edges[to_remove] = 0.0

        # thresholds

        if low_threshold is not None:
            low = thin_edges > low_threshold

            if high_threshold is not None:
                high = thin_edges > high_threshold
                # get black/gray/white only
                thin_edges = low * 0.5 + high * 0.5

                if hysteresis:
                    # get weaks and check if they are high or not
                    weak = (thin_edges == 0.5) * 1
                    weak_is_high = (self.hysteresis(thin_edges) > 1) * weak
                    thin_edges = high * 1 + weak_is_high * 1
            else:
                thin_edges = low * 1


        return blurred, grad_x, grad_y, grad_magnitude, grad_orientation, thin_edges