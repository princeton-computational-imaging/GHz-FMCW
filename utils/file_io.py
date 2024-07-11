#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import re
import h5py
import csv as csv
import pandas as pd
from scipy.signal import savgol_filter
from scipy.fft import fft, ifft
import torch as torch
from os.path import exists



def load_scene_fmcw_data(root_path, model_name):
    path = root_path+"fmcw_signals/"
    filename = "scene_"+model_name+"_signals.csv"

    scene_data = []
    chunksize = 1000
    for df in pd.read_csv(path+filename, dtype=float, header=None, engine='python', chunksize=chunksize):
        df = df.values.tolist()
        for i in range(0, np.shape(df)[0]): # 601
            # smooth data
            smoothed = df[i][:]
            smoothed = savgol_filter(smoothed, 899, 3)
            scene_data.append(smoothed)
    scene_data = np.array(scene_data)
    return scene_data

def load_scene_phase(root_path, model_name):
    path = root_path+"phase/"
    filename = "scene_"+model_name+"_phase.csv"

    with open(path+filename, newline='') as f:
        reader = csv.reader(f)
        scene_phase_2d = list(reader)
    scene_phase_2d = np.array(scene_phase_2d)
    scene_phase_2d = scene_phase_2d.astype(float)

    scene_phase_2d_raw = scene_phase_2d
    return scene_phase_2d_raw

def data_init(scene_data):
    scene_data_test = np.zeros_like(scene_data)
    for i in range(len(scene_data)):
        scene_data_test[i] = fft(scene_data[i])
    
    scene_data_test = torch.from_numpy(scene_data_test).cuda().float()
    n_data, n_features = scene_data_test.shape
    
    return scene_data_test, n_features


def scene_to_2d(scene_depth, v = 100, h = 100):
    scene_depth_2d = np.zeros((v, h))
    for i in range(h):
        for j in range(v):
            scene_depth_2d[j, i] = scene_depth[j + i*v]
    
    return scene_depth_2d

def depth_loader(root_path, model_name):
    path = root_path+"depth/"
    filename = "scene_"+model_name+"_depth.csv"
    filename = path+filename
    file_exists = exists(filename)

    if file_exists == True:
        print("file exists.")
        with open(filename, newline='') as f:
            reader = csv.reader(f)
            depth = list(reader)
        depth = np.array(depth)
        depth = depth.astype(float)
    else:
        print("file does not exist.")
        depth = 0
        
    return depth

def read_pfm(file):
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if header.decode("ascii") == 'PF':
        color = True
    elif header.decode("ascii") == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode("ascii"))
    if dim_match:
        width, height = list(map(int, dim_match.groups()))
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().decode("ascii").rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data, scale

def read_calib_txt(file):
    # hard-coded for middlebury v3 dataset
    with open(file, "r") as f:
        focal = float(f.readline().split(" ")[0][6:])
        _ = f.readline()
        doffs = float(f.readline().split("=")[1])
        baseline = float(f.readline().split("=")[1])
    
    return baseline, focal, doffs

def RGB_to_NIR(img):
    interm = np.maximum(img, 1-img)[...,::-1]
    nir = (interm[..., 0]*0.229 + interm[..., 1]*0.587 + interm[..., 2]*0.114)**(1/0.25)
    return nir

def read_hdf5(file):
    with h5py.File(file, "r") as f:
        key = list(f.keys())[0]
        data = np.array(f[key], dtype=np.float32)
        data = np.nan_to_num(data, nan=0.0)
    return data


# In[ ]:




