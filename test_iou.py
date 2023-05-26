import glob
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms

lbl_transform = transforms.Compose([
    transforms.Resize(size=([128,128])),
    transforms.ToTensor(),
])

def get_F1_score(true, pred):
    """F1"""
    # cast to binary 1st
    true = np.copy(true)
    pred = np.copy(pred)
    true[true > 0] = 1
    pred[pred > 0] = 1
    TP = true * pred
    FP = pred - TP
    FN = true - TP
    precision = TP.sum() / (TP.sum() + FP.sum())
    recall = TP.sum() / (TP.sum() + FN.sum())
    F1 = (2 * precision * recall) / (precision + recall)
    return F1


def get_IoU_score(true, pred):
    """IoU"""
    # cast to binary 1st
    true = np.copy(true)
    pred = np.copy(pred)
    true[true > 0] = 1
    pred[pred > 0] = 1
    TP = true * pred
    FP = pred - TP
    FN = true - TP
    iou = TP.sum() / (TP.sum() + FP.sum() + FN.sum())
    return iou
def get_dice_1(true, pred):
    """Traditional dice."""
    # cast to binary 1st
    true = np.copy(true)
    pred = np.copy(pred)
    true[true > 0] = 1
    pred[pred > 0] = 1
    inter = true * pred
    # 注意这里不是取的并集，而是把两个区域要求和，计算了重叠部分
    denom = true + pred
    return 2.0 * np.sum(inter) / np.sum(denom)
def get_dice_2(true, pred):
    """Ensemble Dice as used in Computational Precision Medicine Challenge."""
    true = np.copy(true)
    pred = np.copy(pred)
    true_id = list(np.unique(true))
    pred_id = list(np.unique(pred))
    # remove background aka id 0，这里是剔除了背景类
    true_id.remove(0)
    pred_id.remove(0)

    total_markup = 0
    total_intersect = 0
    for t in true_id:
        t_mask = np.array(true == t, np.uint8)
        for p in pred_id:
            p_mask = np.array(pred == p, np.uint8)
            intersect = p_mask * t_mask
            if intersect.sum() > 0:
                total_intersect += intersect.sum()
                total_markup += t_mask.sum() + p_mask.sum()
    return 2 * total_intersect / total_markup

def predict(pre_true, lbl_path):
    pre_path = glob.glob(pre_true)
    lbl_path = glob.glob(lbl_path)
    score = [0, 0, 0, 0]
    for i in range(len(pre_path)):
        pre = np.array(lbl_transform(Image.open(pre_path[i])))
        lbl = np.array(lbl_transform(Image.open(lbl_path[i])))
        pre = np.squeeze(pre)
        lbl = np.squeeze(lbl)
        # score[0] = score[0] + get_F1_score(pre, lbl)
        score[1] = score[1] + get_IoU_score(pre, lbl)
        score[2] = score[2] + get_dice_1(pre, lbl)
        # score[3] = score[3] + get_dice_2(pre, lbl)
    for i in range(len(score)):
        score[i] = score[i] / len(pre_path)
    return score

if __name__ == "__main__":

    save_results = "./Data/predict/pre/*.png"
    label_results = "./Data/predict/label/*.png"


    score = predict(save_results, label_results)
    # print("F1_score =", score[0])
    print("Iou_score =", score[1])
    print("dice1_score =", score[2])
    # print("dice2_score =", score[3])

