import sys
sys.path.append("/content/drive/MyDrive/LPRNet/VietNam")  # Đường dẫn chứa load_data.py

from load_data import CHARS, CHARS_DICT, LPRDataLoader
from LPRNet import build_lprnet
# import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import *
from torch import optim
import torch.nn as nn
import numpy as np
import argparse
import torch
import time
import os

# Thêm imports cho confusion matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Chuẩn bị dữ liệu cho CTC Loss
def sparse_tuple_for_ctc(T_length, lengths):
    input_lengths = []
    target_lengths = []

    for ch in lengths:
        input_lengths.append(T_length)
        target_lengths.append(ch)

    return tuple(input_lengths), tuple(target_lengths)

# Điều chỉnh learning rate theo epoch
def adjust_learning_rate(optimizer, cur_epoch, base_lr, lr_schedule):
    # Sets the learning rate
    lr = 0
    for i, e in enumerate(lr_schedule):
        if cur_epoch < e:
            lr = base_lr * (0.1 ** i)
            break
    if lr == 0:
        lr = base_lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr

args = argparse.Namespace(
    max_epoch=20,                                                                                 # Số epoch tối đa để huấn luyện mô hình.
                                                                                                  # Một epoch là một vòng duyệt qua toàn bộ dữ liệu huấn luyện.
                                                                                                  # Giảm tham số này, huấn luyện sẽ kết thúc sớm hơn, nhưng độ chính xác thấp.
                                                                                                  # default = 15
    img_size=[94, 24],                                                                            # Kích thước ảnh (chiều rộng, chiều cao) dùng để resize input.
    train_img_dirs="/content/drive/MyDrive/LPRNet/VietNam/data/image/train",                      # Đường dẫn thư mục ảnh dùng để huấn luyện.
    test_img_dirs="/content/drive/MyDrive/LPRNet/VietNam/data/image/valid",                       # Đường dẫn thư mục ảnh dùng để kiểm tra (validation/test).
    dropout_rate=0.5,                                                                             # Tỷ lệ dropout (xác suất loại bỏ neuron ngẫu nhiên trong quá trình huấn luyện), giúp tránh overfitting.
    learning_rate=0.001,                                                                          # Tốc độ học của mô hình. Giá trị này càng nhỏ, mô hình học càng chậm nhưng ổn định.
    lpr_max_len=9,                                                                                # Chiều dài tối đa của chuỗi ký tự biển số xe (số lượng ký tự trong biển số).
    train_batch_size=32,                                                                          # Số ảnh mỗi batch trong quá trình huấn luyện.
    test_batch_size=20,                                                                           # Số ảnh mỗi batch trong quá trình kiểm tra.
    phase_train=True,                                                                             # Cờ cho biết mô hình đang trong chế độ huấn luyện (True) hay chỉ inference (False).
                                                                                                  # Sử dụng CPU để huấn luyện. Nếu có GPU, nên đặt thành True để huấn luyện nhanh hơn.
    num_workers=2,                                                                                # Số luồng dùng để load dữ liệu song song.
    cuda=False,                                                                                   # dùng CPU trong Colab, sửa True nếu bạn dùng GPU
    resume_epoch=0,                                                                               # Epoch bắt đầu lại quá trình huấn luyện (nếu tiếp tục từ checkpoint đã lưu).
    save_interval=2000,                                                                           # Số bước (iteration) giữa mỗi lần lưu model.
    test_interval=2000,                                                                           # Số bước giữa mỗi lần kiểm tra mô hình trên tập test/validation.
    momentum=0.9,                                                                                 # Tham số dùng trong các optimizer như SGD để giúp tăng tốc độ hội tụ và tránh local minima.
    weight_decay=2e-5,                                                                            # Tham số regularization để tránh overfitting, thường dùng trong optimizer.
    lr_schedule=[10, 20, 40, 60, 80, 100],                                                        # Epochs mà tại đó learning rate sẽ được điều chỉnh (thường giảm đi).
    save_folder='/content/drive/MyDrive/LPRNet/VietNam/weights/',                                 # Thư mục lưu trọng số của mô hình.
    # pretrained_model='/content/drive/MyDrive/LPRNet/VietNam/weights/Final_LPRNet_model.pth'     # Đường dẫn model đã được huấn luyện sẵn (nếu muốn tiếp tục từ đó). Nếu không có sẵn pretrained_model, đặt ''
    pretrained_model=''
)

# Xử lý batch data từ DataLoader
def collate_fn(batch):
    imgs = []
    labels = []
    lengths = []
    for _, sample in enumerate(batch):
        img, label, length = sample
        imgs.append(torch.from_numpy(img))
        labels.extend(label)
        lengths.append(length)
    # labels = np.asarray(labels).flatten().astype(int)
    labels = np.asarray(labels).flatten().astype(np.float32)

    return (torch.stack(imgs, 0), torch.from_numpy(labels), lengths)

# Đánh giá accuracy bằng greedy decoding
def Greedy_Decode_Eval(Net, datasets, args, device):
    # TestNet = Net.eval()
    epoch_size = len(datasets) // args.test_batch_size
    batch_iterator = iter(DataLoader(datasets, args.test_batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn))

    Tp = 0
    Tn_1 = 0
    Tn_2 = 0
    t1 = time.time()
    for i in range(epoch_size):
        # load train data
        images, labels, lengths = next(batch_iterator)
        start = 0
        targets = []
        for length in lengths:
            label = labels[start:start+length]
            targets.append(label)
            start += length
        # targets = np.array([el.numpy() for el in targets])
        targets = [el.numpy() for el in targets]  # Keep it as a list of NumPy arrays

        if args.cuda:
            # images = Variable(images.cuda())
            images = Variable(images.to(device))
        else:
            images = Variable(images)

        # forward
        prebs = Net(images)
        # greedy decode
        prebs = prebs.cpu().detach().numpy()
        preb_labels = list()
        for i in range(prebs.shape[0]):
            preb = prebs[i, :, :]
            preb_label = list()
            for j in range(preb.shape[1]):
                preb_label.append(np.argmax(preb[:, j], axis=0))
            no_repeat_blank_label = list()
            pre_c = preb_label[0]
            if pre_c != len(CHARS) - 1:
                no_repeat_blank_label.append(pre_c)
            for c in preb_label: # dropout repeate label and blank label
                if (pre_c == c) or (c == len(CHARS) - 1):
                    if c == len(CHARS) - 1:
                        pre_c = c
                    continue
                no_repeat_blank_label.append(c)
                pre_c = c
            preb_labels.append(no_repeat_blank_label)
        for i, label in enumerate(preb_labels):
            if len(label) != len(targets[i]):
                Tn_1 += 1
                continue
            if (np.asarray(targets[i]) == np.asarray(label)).all():
                Tp += 1
            else:
                Tn_2 += 1

    Acc = Tp * 1.0 / (Tp + Tn_1 + Tn_2)
    print("[Info] Test Accuracy: {} [{}:{}:{}:{}]".format(Acc, Tp, Tn_1, Tn_2, (Tp+Tn_1+Tn_2)))
    t2 = time.time()
    print("[Info] Test Speed: {}s 1/{}]".format((t2 - t1) / len(datasets), len(datasets)))

# ===== THÊM CÁC FUNCTION CHO CONFUSION MATRIX =====

# Thu thập data cho confusion matrix
def collect_predictions_and_labels(Net, datasets, args, device):
    # Thu thập predictions và true labels để tạo confusion matrix
    Net.eval()
    epoch_size = len(datasets) // args.test_batch_size
    batch_iterator = iter(DataLoader(datasets, args.test_batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn))

    all_predictions = []
    all_true_labels = []

    print("Đang thu thập predictions cho confusion matrix...")

    for i in range(epoch_size):
        # Load test data
        images, labels, lengths = next(batch_iterator)
        start = 0
        targets = []
        for length in lengths:
            label = labels[start:start+length]
            targets.append(label)
            start += length
        targets = [el.numpy() for el in targets]

        if args.cuda:
            images = Variable(images.to(device))
        else:
            images = Variable(images)

        # Forward pass
        prebs = Net(images)
        prebs = prebs.cpu().detach().numpy()

        # Greedy decode
        preb_labels = list()
        for j in range(prebs.shape[0]):
            preb = prebs[j, :, :]
            preb_label = list()
            for k in range(preb.shape[1]):
                preb_label.append(np.argmax(preb[:, k], axis=0))

            # Remove repeats and blanks
            no_repeat_blank_label = list()
            pre_c = preb_label[0]
            if pre_c != len(CHARS) - 1:
                no_repeat_blank_label.append(pre_c)
            for c in preb_label:
                if (pre_c == c) or (c == len(CHARS) - 1):
                    if c == len(CHARS) - 1:
                        pre_c = c
                    continue
                no_repeat_blank_label.append(c)
                pre_c = c
            preb_labels.append(no_repeat_blank_label)

        # Collect character-level predictions and labels
        for pred, true in zip(preb_labels, targets):
            # Chỉ lấy các ký tự có thật (không padding)
            min_len = min(len(pred), len(true))
            all_predictions.extend(pred[:min_len])
            all_true_labels.extend(true[:min_len].astype(int))

    return all_predictions, all_true_labels

# Vẽ và phân tích confusion matrix
def plot_confusion_matrix(y_true, y_pred, save_path=None):
    # Tạo và hiển thị confusion matrix
    if not y_true or not y_pred:
        print("Không có dữ liệu để tạo confusion matrix")
        return None

    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Get unique labels that actually appear in the data
    unique_labels = sorted(list(set(y_true + y_pred)))
    char_labels = [CHARS[i] if i < len(CHARS) else f'UNK_{i}' for i in unique_labels]

    # Plot confusion matrix
    plt.figure(figsize=(15, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=char_labels, yticklabels=char_labels,
                cbar_kws={'label': 'Số lượng'})
    plt.title('Confusion Matrix - Nhận dạng ký tự biển số xe', fontsize=16, pad=20)
    plt.xlabel('Ký tự dự đoán', fontsize=12)
    plt.ylabel('Ký tự thực tế', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix đã được lưu tại: {save_path}")

    plt.show()

    # Print classification report
    print("\n" + "="*60)
    print("BÁO CÁO PHÂN LOẠI CHI TIẾT:")
    print("="*60)
    try:
        report = classification_report(y_true, y_pred,
                                     target_names=char_labels,
                                     zero_division=0,
                                     digits=4)
        print(report)
    except Exception as e:
        print(f"Không thể tạo classification report: {e}")

    return cm

# Phân tích toàn diện hiệu suất model
def analyze_model_performance(Net, test_dataset, args, device):
    # Phân tích hiệu suất model và tạo confusion matrix
    print("\n" + "="*60)
    print("PHÂN TÍCH HIỆU SUẤT MODEL VỚI CONFUSION MATRIX")
    print("="*60)

    # Collect predictions and true labels
    predictions, true_labels = collect_predictions_and_labels(Net, test_dataset, args, device)

    if not predictions or not true_labels:
        print("Không thu thập được predictions. Không thể tạo confusion matrix.")
        return

    print(f"Đã thu thập {len(predictions)} dự đoán ký tự")

    # Create confusion matrix
    save_path = os.path.join(args.save_folder, 'confusion_matrix.png')
    cm = plot_confusion_matrix(true_labels, predictions, save_path)

    # Calculate character-level accuracy
    correct = sum(1 for t, p in zip(true_labels, predictions) if t == p)
    total = len(true_labels)
    char_accuracy = correct / total if total > 0 else 0

    print(f"\nĐộ chính xác ở mức ký tự: {char_accuracy:.4f} ({correct}/{total})")

    # Most confused characters
    if cm is not None and cm.size > 0:
        print("\nCác cặp ký tự bị nhầm lẫn nhiều nhất:")
        unique_labels = sorted(list(set(true_labels + predictions)))

        confusion_pairs = []
        for i, true_idx in enumerate(unique_labels):
            for j, pred_idx in enumerate(unique_labels):
                if i != j and i < len(cm) and j < len(cm[0]) and cm[i, j] > 0:
                    true_char = CHARS[true_idx] if true_idx < len(CHARS) else f'UNK_{true_idx}'
                    pred_char = CHARS[pred_idx] if pred_idx < len(CHARS) else f'UNK_{pred_idx}'
                    confusion_pairs.append((cm[i, j], true_char, pred_char))

        confusion_pairs.sort(reverse=True)
        for count, true_char, pred_char in confusion_pairs[:10]:
            print(f"  '{true_char}' -> '{pred_char}': {count} lần")

    # Character frequency analysis
    print(f"\nPhân tích tần suất ký tự:")
    char_freq = {}
    for label in true_labels:
        char = CHARS[label] if label < len(CHARS) else f'UNK_{label}'
        char_freq[char] = char_freq.get(char, 0) + 1

    print("Top 10 ký tự xuất hiện nhiều nhất:")
    sorted_chars = sorted(char_freq.items(), key=lambda x: x[1], reverse=True)
    for char, freq in sorted_chars[:10]:
        print(f"  '{char}': {freq} lần")

# Hàm điều chỉnh và điều phối toàn bộ quá trình training
def train():
    # args = get_parser()

    T_length = 18 # args.lpr_max_len
    epoch = 0 + args.resume_epoch
    loss_val = 0

    if not os.path.exists(args.save_folder):
        os.mkdir(args.save_folder)

    lprnet = build_lprnet(lpr_max_len=args.lpr_max_len, phase=args.phase_train, class_num=len(CHARS), dropout_rate=args.dropout_rate)
    # device = torch.device("cuda:0" if args.cuda else "cpu")
    device = torch.device("cpu")
    lprnet.to(device)
    print("Successful to build network!")

    # load pretrained model
    if args.pretrained_model:
        lprnet.load_state_dict(torch.load(args.pretrained_model))
        print("Load pretrained model successful!")
    else:
        def xavier(param):
            nn.init.xavier_uniform(param)

        def weights_init(m):
            for key in m.state_dict():
                if key.split('.')[-1] == 'weight':
                    if 'conv' in key:
                        nn.init.kaiming_normal_(m.state_dict()[key], mode='fan_out')
                    if 'bn' in key:
                        m.state_dict()[key][...] = xavier(1)
                elif key.split('.')[-1] == 'bias':
                    m.state_dict()[key][...] = 0.01

        lprnet.backbone.apply(weights_init)
        lprnet.container.apply(weights_init)
        print("Initial net weights successful!")

    # define optimizer
    # optimizer = optim.SGD(lprnet.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer = optim.RMSprop(lprnet.parameters(), lr=args.learning_rate, alpha = 0.9, eps=1e-08, momentum=args.momentum, weight_decay=args.weight_decay)
    train_img_dirs = os.path.expanduser(args.train_img_dirs)
    test_img_dirs = os.path.expanduser(args.test_img_dirs)
    train_dataset = LPRDataLoader(train_img_dirs.split(','), args.img_size, args.lpr_max_len)
    test_dataset = LPRDataLoader(test_img_dirs.split(','), args.img_size, args.lpr_max_len)

    epoch_size = len(train_dataset) // args.train_batch_size
    max_iter = args.max_epoch * epoch_size
    print("DEBUG: epoch_size =", epoch_size)
    print("DEBUG: max_iter =", max_iter)

    ctc_loss = nn.CTCLoss(blank=len(CHARS)-1, reduction='mean') # reduction: 'none' | 'mean' | 'sum'

    if args.resume_epoch > 0:
        start_iter = args.resume_epoch * epoch_size
    else:
        start_iter = 0

    for iteration in range(start_iter, max_iter):
        if iteration % epoch_size == 0:
            # create batch iterator
            batch_iterator = iter(DataLoader(train_dataset, args.train_batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn))
            loss_val = 0
            epoch += 1

        if iteration !=0 and iteration % args.save_interval == 0:
            torch.save(lprnet.state_dict(), args.save_folder + 'LPRNet_' + '_iteration_' + repr(iteration) + '.pth')

        if (iteration + 1) % args.test_interval == 0:
            Greedy_Decode_Eval(lprnet, test_dataset, args, device)
            # lprnet.train() # should be switch to train mode

        start_time = time.time()
        # load train data
        images, labels, lengths = next(batch_iterator)
        # labels = np.array([el.numpy() for el in labels]).T
        # print(labels)
        # get ctc parameters
        input_lengths, target_lengths = sparse_tuple_for_ctc(T_length, lengths)
        # update lr
        lr = adjust_learning_rate(optimizer, epoch, args.learning_rate, args.lr_schedule)

        if args.cuda:
            images = Variable(images, requires_grad=False).to(device)
            labels = Variable(labels, requires_grad=False).to(device)
        else:
            images = Variable(images, requires_grad=False)
            labels = Variable(labels, requires_grad=False)

        # forward
        logits = lprnet(images)
        log_probs = logits.permute(2, 0, 1) # for ctc loss: T x N x C
        # print(labels.shape)
        log_probs = log_probs.log_softmax(2).requires_grad_()
        # log_probs = log_probs.detach().requires_grad_()
        # print(log_probs.shape)
        # backprop
        optimizer.zero_grad()
        loss = ctc_loss(log_probs, labels, input_lengths=input_lengths, target_lengths=target_lengths)
        if loss.item() == np.inf:
            continue
        loss.backward()
        optimizer.step()
        loss_val += loss.item()
        end_time = time.time()
        if iteration % 20 == 0:
            print('Epoch:' + repr(epoch) + ' || epochiter: ' + repr(iteration % epoch_size) + '/' + repr(epoch_size) + '|| Totel iter ' + repr(iteration) + ' || Loss: %.4f||' % (loss.item()) + 'Batch time: %.4f sec. ||' % (end_time - start_time) + 'LR: %.4f' % (lr))

    # ===== PHẦN ĐÁNH GIÁ CUỐI CÙNG VỚI CONFUSION MATRIX =====
    print("\n" + "="*60)
    print("ĐÁNH GIÁ MODEL CUỐI CÙNG VỚI CONFUSION MATRIX")
    print("="*60)

    # Final accuracy test (giữ nguyên)
    print("Độ chính xác test cuối cùng:")
    Greedy_Decode_Eval(lprnet, test_dataset, args, device)

    # Thêm confusion matrix analysis
    analyze_model_performance(lprnet, test_dataset, args, device)

    # Save final model (giữ nguyên)
    print(f"\nĐang lưu model cuối cùng tại: {args.save_folder}Final_LPRNet_model.pth")
    torch.save(lprnet.state_dict(), args.save_folder + 'Final_LPRNet_model.pth')
    print("Model đã được lưu thành công!")

if __name__ == "__main__":
    train()
