import glob
import random

def write_txt(txt_path, txt_name, txt):
    f1 = open(txt_path + "/" + txt_name, 'w')
    for j in txt:
        f1.write(j + '\n')
    f1.close()

path = "D:/Prostate/Data/image/*.npy"
image_name = glob.glob(path)
txt_list = []

for i in range(len(image_name)):
    label_name = image_name[i].replace("image", "mask")
    txt_list.append(image_name[i] + " " + label_name)

random.shuffle(txt_list)
ll = len(txt_list)
train_txt = txt_list[0:int(0.8 * ll)]
test_txt = txt_list[int(0.8 * ll):ll]



write_txt("D:/Prostate/Data", "train.txt", train_txt)
write_txt("D:/Prostate/Data", "test.txt", test_txt)
