import matplotlib.pyplot as plt
import numpy as np

path = "D:/code/python/Prostate/Data/mask/0_12.npy"
a = np.load(path).astype("uint8")
print(np.unique(a))
plt.imshow(a)