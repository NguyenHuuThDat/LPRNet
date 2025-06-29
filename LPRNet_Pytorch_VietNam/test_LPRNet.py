# Bổ sung thư mục chứa file load_data.py và LPRNet.py
import sys
sys.path.append("/content/drive/MyDrive/LPRNet/VietNam")

# Import thư viện và hàm cần thiết
from load_data import CHARS, CHARS_DICT, LPRDataLoader
from PIL import Image, ImageDraw, ImageFont
from LPRNet import build_lprnet
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import optim
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import argparse
import torch
import time
import cv2
import os

# Khai báo args thủ công, tránh dùng argparse.parse_args()
args = argparse.Namespace(
    img_size=[94, 24],
    test_img_dirs="/content/drive/MyDrive/LPRNet/VietNam/data/image/valid",
    dropout_rate=0.0,
    lpr_max_len=9,
    test_batch_size=20,
    phase_train=False,
    num_workers=0,
    cuda=torch.cuda.is_available(),
    show=False,
    pretrained_model="/content/drive/MyDrive/LPRNet/VietNam/weights/Final_LPRNet_model.pth"
)

def collate_fn(batch):
    imgs = []
    labels = []
    lengths = []
    for _, sample in enumerate(batch):
        img, label, length = sample
        imgs.append(torch.from_numpy(img))
        labels.extend(label)
        lengths.append(length)
    labels = np.asarray(labels).flatten().astype(np.float32)
    return (torch.stack(imgs, 0), torch.from_numpy(labels), lengths)

def test():
    device = torch.device("cuda" if args.cuda else "cpu")

    lprnet = build_lprnet(
        lpr_max_len=args.lpr_max_len,
        phase=args.phase_train,
        class_num=len(CHARS),
        dropout_rate=args.dropout_rate
    ).to(device)

    print("Model built successfully")

    if args.pretrained_model and os.path.exists(args.pretrained_model):
        lprnet.load_state_dict(torch.load(args.pretrained_model, map_location=device))
        print("Loaded pretrained model successfully")
    else:
        print("Error: Pretrained model not found!")
        return

    test_img_dirs = os.path.expanduser(args.test_img_dirs)
    test_dataset = LPRDataLoader(test_img_dirs.split(','), args.img_size, args.lpr_max_len)
    if len(test_dataset) == 0:
        print("Error: Không có ảnh trong thư mục test!")
        return

    try:
        Greedy_Decode_Eval(lprnet, test_dataset, args, device)
    finally:
        cv2.destroyAllWindows()

def Greedy_Decode_Eval(Net, datasets, args, device):
    epoch_size = len(datasets) // args.test_batch_size
    batch_iterator = iter(DataLoader(datasets, args.test_batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn))

    Tp = 0
    Tn_1 = 0
    Tn_2 = 0
    total_levenshtein_distance = 0
    t1 = time.time()
    inference_times = []

    sample_results = []

    for i in range(epoch_size):
        start_time = time.time()
        images, labels, lengths = next(batch_iterator)
        start = 0
        targets = []
        for length in lengths:
            label = labels[start:start+length]
            targets.append(label)
            start += length
        imgs = images.numpy().copy()

        if args.cuda:
            images = images.to(device)
        else:
            images = Variable(images)

        prebs = Net(images)
        prebs = prebs.cpu().detach().numpy()
        preb_labels = list()
        for i in range(prebs.shape[0]):
            preb = prebs[i, :, :]
            preb_label = [np.argmax(preb[:, j], axis=0) for j in range(preb.shape[1])]
            no_repeat_blank_label = []
            pre_c = preb_label[0]
            if pre_c != len(CHARS) - 1:
                no_repeat_blank_label.append(pre_c)
            for c in preb_label:
                if (pre_c == c) or (c == len(CHARS) - 1):
                    pre_c = c
                    continue
                no_repeat_blank_label.append(c)
                pre_c = c
            preb_labels.append(no_repeat_blank_label)

        for i, label in enumerate(preb_labels):
            if len(label) != len(targets[i]):
                Tn_1 += 1
                continue
            label = np.asarray(label)
            target_i = np.asarray(targets[i])
            if (label == target_i).all():
                Tp += 1
            else:
                Tn_2 += 1

            pred_str = ''.join([CHARS[idx] for idx in label])
            target_str = ''.join([CHARS[int(idx)] for idx in targets[i]])
            levenshtein_distance_value = levenshtein_distance(pred_str, target_str)
            total_levenshtein_distance += levenshtein_distance_value

            # if len(sample_results) < 3:
            sample_results.append((imgs[i], pred_str, target_str, levenshtein_distance_value))

        end_time = time.time()
        inference_time = end_time - start_time
        inference_times.append(inference_time)

    total_samples = Tp + Tn_1 + Tn_2
    Acc = Tp / total_samples if total_samples > 0 else 0
    mean_levenshtein_distance = total_levenshtein_distance / total_samples if total_samples > 0 else 0
    avg_inference_time = np.mean(inference_times) / args.test_batch_size

    print("[Info] Test Accuracy: {:.4f} [TP:{} Tn_1:{} Tn_2:{} Total:{}]".format(Acc, Tp, Tn_1, Tn_2, total_samples))
    print("[Info] Mean Levenshtein Distance: {:.4f}".format(mean_levenshtein_distance))
    print("[Info] Avg Inference Time per Sample: {:.6f}s".format(avg_inference_time))

    print("\nSample Predictions and Levenshtein Distances:")
    for img, pred_str, target_str, lev in sample_results:
        show(img, pred_str, target_str)
        print("Predicted: {}, Target: {}, Levenshtein Distance: {}".format(pred_str, target_str, lev))

def show(img, pred_str, target_str, save_dir="/content/drive/MyDrive/LPRNet/VietNam/output"):
    img = np.transpose(img, (1, 2, 0))
    img = (img * 128.0 + 127.5).astype(np.uint8)

    flag = "T" if pred_str == target_str else "F"
    label_text = f"{target_str}_{flag}_{pred_str}"
    img = cv2ImgAddText(img, label_text, (5, 5))

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    filepath = os.path.join(save_dir, f"{label_text}.jpg")
    cv2.imwrite(filepath, img)
    print(f"Đã lưu ảnh: {filepath}")

def cv2ImgAddText(img, text, pos, textColor=(255, 0, 0), textSize=20):
    if isinstance(img, np.ndarray):
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("Arial.ttf", textSize)
    except:
        font = ImageFont.load_default()
    draw.text(pos, text, textColor, font=font)
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

def levenshtein_distance(a, b):
    n, m = len(a), len(b)
    if n > m:
        a, b = b, a
        n, m = m, n
    current_row = range(n + 1)
    for i in range(1, m + 1):
        previous_row, current_row = current_row, [i] + [0] * n
        for j in range(1, n + 1):
            add, delete, change = previous_row[j] + 1, current_row[j - 1] + 1, previous_row[j - 1]
            if a[j - 1] != b[i - 1]:
                change += 1
            current_row[j] = min(add, delete, change)
    return current_row[n]

# Gọi test() trực tiếp khi chạy Colab
test()
