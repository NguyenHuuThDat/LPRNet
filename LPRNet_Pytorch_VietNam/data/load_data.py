
from torch.utils.data import Dataset
from imutils import paths
import numpy as np
import random
import cv2
import os

# Danh sách tất cả các ký tự có thể xuất hiện trong biển số xe
CHARS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
         'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
         'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
         'W', 'X', 'Y', 'Z', 'I', 'O', '-']

# Từ điển ánh xạ từ ký tự sang chỉ số tương ứng
CHARS_DICT = {char: i for i, char in enumerate(CHARS)}

class LPRDataLoader(Dataset):
    # Khởi thao đối tượng với các tham số:
    # img_dir: Danh sách các thư mục chứa ảnh biển số
    # imgSize: Kích thước ảnh đầu vào
    # lpr_max_len: Độ dài tối đa của biển số
    # PreprocFun: Hàm xử lý ảnh
    def __init__(self, img_dir, imgSize, lpr_max_len, PreprocFun=None):
        self.img_dir = img_dir                                                      # Lưu danh sách thư mục ảnh
        self.img_paths = [el for dir in img_dir for el in paths.list_images(dir)]   # Tạo danh sách đường dẫn đến tất cả các file ảnh trong các thư mục
        random.shuffle(self.img_paths)                                              # Xáo trộn ngẫu nhiên danh sách ảnh
        self.img_size = imgSize                                                     # Lưu kích thước ảnh đầu vào
        self.lpr_max_len = lpr_max_len                                              # Lưu độ dài tối đa của biển số
        self.PreprocFun = PreprocFun if PreprocFun is not None else self.transform  # Gán hàm tiền xử lý, nếu không được cung cấp thì dùng hàm transform mặc định

    # Trả về số lượng ảnh, giúp Dataloader biết khi nào dừng 1 epoch
    def __len__(self):
        return len(self.img_paths)

    # Lấy 1 mẫu dữ liệu tại vị trí index
    # Đọc ảnh bằng cv2.imread()
    # Nếu kích thước ảnh khác với self.img_size => resize
    # Áp dụng hàm tiền xử lý self.PreprocFun lên ảnh
    # Trích xuất nhãn từ tên file
    # Chuyển đổi nhãn từ chuỗi ký tự sang danh sách chỉ số
    # Kiểm tra độ dài nhãn có nằm trong khoảng [7, 9] không. Nếu không, in thông báo lỗi và trả về None
    # Trả về tuple bao gồm ảnh đã được xử lý, nhãn dưới dạng danh sách chỉ số, độ dài nhãn
    def __getitem__(self, index):
        filename = self.img_paths[index]
        image = cv2.imread(filename)
        image = cv2.resize(image, self.img_size) if (image.shape[0], image.shape[1]) != self.img_size else image
        image = self.PreprocFun(image)

        basename = os.path.basename(filename).split('_')[0].split('.')[0]
        label = [CHARS_DICT[char] for char in basename if char in CHARS_DICT]

        if not 7 <= len(label) <= 9:
            print(f"Invalid label length for {basename}")
            return None

        return image, label, len(label)

    # Tiền xử lý ảnh
    # Chuyển đổi ảnh sang KDL float32
    # Chuẩn hóa giá trị pixel bằng cách trừ 127.5 => đưa về khoảng [-127.5, 127.5]
    # Nhân với 1/128 để đưa về khoảng [-0.996, 0.996]
    def transform(self, img):
        img = img.astype('float32')
        img -= 127.5
        img *= 0.0078125
        img = np.transpose(img, (2, 0, 1))
        return img

    def check(self, label):
        return True
