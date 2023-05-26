import torch
import torch.nn as nn
import torch.optim as optim
from DataSet import ProState
from torch.utils.data import DataLoader
from model import AttUNet
import matplotlib.pyplot as plt
from Tool import transform
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

train_transform = transform.Compose(
        [
            transform.RandomResizedCrop(128, (0.5, 2.0)),
            transform.RandomHorizontalFlip(),
            transform.ToTensor(),
            transform.Normalize(mean=[0.485, ], std=[0.229, ]),
        ]
    )

def train(net, device, train_txt):
    y = []
    train_data = ProState(train_txt, train_transform)
    train_dataset = DataLoader(train_data, batch_size=6, shuffle=True)

    # 定义RMSprop算法
    optimizer = optim.RMSprop(net.parameters(), lr=0.000005, weight_decay=1e-8, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.75, min_lr=0.0, patience=10)

    # 定义Loss算法
    criterion = nn.CrossEntropyLoss(reduce="mean")
    best_loss = float('inf')
    # 训练epochs次
    train_loss = 0
    for epoch in range(1, 40):
        # 训练模式
        net.train()
        # 按照batch_size开始训练
        for image, label in train_dataset:
            optimizer.zero_grad()  # 梯度置零
            # 将数据拷贝到device中
            image = image.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.float32)
            # 使用网络参数，输出预测结果
            pred = net(image)
            label = torch.squeeze(label, dim=1)  # torch自带函数删除第二维度
            # 计算loss
            loss = criterion(pred, label.long())
            train_loss += loss.item()
            y.append(loss.item())
            print("epoch =", epoch, 'Loss/train =', loss.item())
            # 更新参数
            loss.backward()
            optimizer.step()

            # 保存loss值最小的网络参数
            if loss < best_loss:
                best_loss = loss
                torch.save(net.state_dict(), 'model_save/best_model.pth')
        if epoch % 5 == 0:
            scheduler.step(train_loss)
            train_loss = 0

    return y


if __name__ == "__main__":
    torch.cuda.empty_cache()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    unet = AttUNet(in_channel=1, out_channel=2)
    unet.to(device=device)
    train_txt = "Data/train1.txt"
    y = train(unet, device, train_txt)
    plt.plot(y)
    plt.savefig('model_save/loss.jpg')






