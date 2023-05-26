import os
import glob
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt

def read_raw(path):
    mhd = sitk.ReadImage(path)
    scan = sitk.GetArrayFromImage(mhd)
    return scan

path = "Prostate/Data/img/*.mhd"
image_path = glob.glob(path)
label_path = glob.glob(path.replace("img", "lbl"))
save_path = "F:/project/Project_file/Prostate/Data/"

for i in range(len(image_path)):
    img = read_raw(image_path[i])
    lbl = read_raw(label_path[i])
    x = img.shape
    for j in range(x[0]):
        a = np.unique(lbl[j, :, :])
        if len(a) == 2:
            np.save(save_path + "image/" + str(i) + "_" + str(j) + ".npy", img[j, :, :])
            np.save(save_path + "mask/" + str(i) + "_" + str(j) + ".npy", lbl[j, :, :])





