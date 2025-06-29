import sys
sys.path.append("/content/drive/MyDrive/LPRNet/VietNam")  # Đường dẫn chứa load_data.py

from LPRNet import build_lprnet
from torchsummary import summary
import argparse
from load_data import CHARS
import torch

args = argparse.Namespace(
    max_epoch=20,
    img_size=[94, 24],
    train_img_dirs="/content/drive/MyDrive/LPRNet/VietNam/data/image/train",
    test_img_dirs="/content/drive/MyDrive/LPRNet/VietNam/data/image/valid",
    dropout_rate=0.5,
    learning_rate=0.001,
    lpr_max_len=9,
    train_batch_size=32,
    test_batch_size=20,
    phase_train=True,
    num_workers=2,
    cuda=torch.cuda.is_available(),
    resume_epoch=0,
    save_interval=2000,
    test_interval=2000,
    momentum=0.9,
    weight_decay=2e-5,
    lr_schedule=[10, 20, 40, 60, 80, 100],
    save_folder='/content/drive/MyDrive/LPRNet/VietNam/weights',
    # pretrained_model='/content/drive/MyDrive/LPRNet/VietNam/weights/Final_LPRNet_model.pth'
    pretrained_model=''
)

# args = get_parser()

lprnet = build_lprnet(lpr_max_len=args.lpr_max_len, phase=args.phase_train, class_num=len(CHARS), dropout_rate=args.dropout_rate)

# Chuyển mô hình lên GPU nếu args.cuda là True
# if args.cuda:
#     lprnet = lprnet.cuda()

# summary(lprnet, input_size=(3, 24, 94), device="cuda" if args.cuda else "cpu")

# Chuyển lên GPU nếu có
device = "cuda" if args.cuda else "cpu"
lprnet = lprnet.to(device)

# Hiển thị kiến trúc model
summary(lprnet, input_size=(3, 24, 94), device=device)