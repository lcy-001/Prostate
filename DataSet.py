import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from Tool import transform


train_transform = transform.Compose(
        [
            transform.RandomResizedCrop(512,  (0.5, 2.0)),
            transform.RandomHorizontalFlip(),
            transform.ToTensor(),
            transform.Normalize(mean=[0.485], std=[0.229]),
        ]
    )

class ProState(Dataset):
    def __init__(self, txt_path, transform):
        with open(txt_path, "r") as f:
            self.file_names = [x[:-1].split(' ') for x in f.readlines()]

        self.transform = transform

    def __getitem__(self, index):
        image = Image.fromarray(np.load(self.file_names[index][0]).astype("uint8"))
        label = Image.fromarray(np.load(self.file_names[index][1]).astype("uint8"))

        image, label = self.transform(image, label)

        return image, label

    def __len__(self):
        return len(self.file_names)

if __name__ == "__main__":

    train_txt = "F:/project/Project_file/Prostate/Data/train.txt"
    train_data = ProState(train_txt, train_transform)
    train_dataset = DataLoader(train_data, batch_size=2, shuffle=True)
    for image, label in train_dataset:
        print(np.unique(label))



