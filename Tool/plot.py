import glob

import matplotlib.pyplot as plt
import numpy as np
import glob
path = glob.glob("Prostate/Data/image/*.npy")
a = []
for i in range(len(path)):
    image = np.load(path[i]).astype("uint8")
    #label = np.load(path.replace("image", "mask")).astype("uint8")
    a.append(image.max())
    a.append(image.min())

a = np.unique(np.array(a))
print(a)





a = np.unique(image)
print(a)
plt.figure(1)
plt.imshow(image)
plt.figure(2)
plt.imshow(label)

