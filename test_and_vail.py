import numpy as np
import torch
from model import AttUNet
from DataSet import ProState
from torch.utils.data import DataLoader
from Tool import transform
import matplotlib.pyplot as plt
from PIL import Image

palette = [(0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128), (0, 128, 128),
           (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128), (192, 0, 128),
           (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128),
           (128, 64, 12)]
val_transform = transform.Compose(
    [
        transform.Resize(size=128),
        transform.CenterCrop(size=128),
        transform.ToTensor(),
        transform.Normalize(mean=[0.485], std=[0.229]),
    ]
)


def cam_mask(mask, palette):
    seg_img = np.zeros((np.shape(mask)[0], np.shape(mask)[1], 3))
    for c in [0, 1]:
        seg_img[:, :, 0] += ((mask[:, :] == c) * (palette[c][0])).astype('uint8')
        seg_img[:, :, 1] += ((mask[:, :] == c) * (palette[c][1])).astype('uint8')
        seg_img[:, :, 2] += ((mask[:, :] == c) * (palette[c][2])).astype('uint8')
    colorized_mask = Image.fromarray(np.uint8(seg_img))
    return colorized_mask

def Test(device, n_channels, n_classes, model_path, save_path, test_txt):
    test_data = ProState(test_txt, val_transform)
    test_dataset = DataLoader(test_data, batch_size=1, shuffle=True)

    net = AttUNet(in_channel=n_channels, out_channel=n_classes)
    # 将网络拷贝到deivce中
    net.to(device=device)
    # 加载模型参数
    net.load_state_dict(torch.load(model_path, map_location=device))
    # 测试模式
    net.eval()
    for i, (image, label) in enumerate(test_dataset):
        image = image.to(device=device, dtype=torch.float32)
        with torch.no_grad():
            pred = net(image)

        pred = pred.argmax(dim=1)[0].cpu().numpy().astype("uint8")
        label = label.numpy()[0]

        pred = cam_mask(pred, palette)
        label = cam_mask(label, palette)
        pred.save(save_path + "pre/" + str(i) + ".png")
        label.save(save_path + "label/" + str(i) + ".png")









if __name__ == "__main__":
    torch.cuda.empty_cache()
    save_path = "Data/predict/"
    model_path = "model_save/best_model.pth"
    test_txt = "Data/test.txt"


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 选择设备，有cuda用cuda，没有就用cpu
    Test(device, 1, 2, model_path, save_path, test_txt)
    print("预测完成")
