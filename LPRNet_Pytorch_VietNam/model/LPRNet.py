
import torch.nn as nn
import torch

# Trích xuất đặc trưng hiệu quả với ít tham số hơn. Sử dụng kiến trúc bottleneck để giảm số lượng tham số
class small_basic_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(small_basic_block, self).__init__()
        self.block = nn.Sequential(
            # Giảm số kênh từ ch_in xuống ch_out // 4. Đây là bước nén thông tin
            nn.Conv2d(ch_in, ch_out // 4, kernel_size=1),
            nn.ReLU(),
            # Convolution theo chiều dọc (3x1). Trích xuất đặc trưng theo chiều cao của ảnh. Padding (1, 0) để giữ nguyên kích thước
            nn.Conv2d(ch_out // 4, ch_out // 4, kernel_size=(3, 1), padding=(1, 0)),
            nn.ReLU(),
            # Convolution theo chiều ngang (1x3). Trích xuất đặc trưng theo chiều rộng của ảnh. Padding (0, 1) để giữ nguyên kích thước
            nn.Conv2d(ch_out // 4, ch_out // 4, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU(),
            # Mở rộng số kênh từ ch_out // 4  lên ch_out. Đây là bước giải nén thông tin
            nn.Conv2d(ch_out // 4, ch_out, kernel_size=1),
        )
    def forward(self, x):
        return self.block(x)

# Mạng thực hiện nhận dạng biển số
class LPRNet(nn.Module):
    # lpr_max_len: độ dài tối đa của biển số
    # phase: chế độ train/eval
    # class_num: số lượng ký tự có thể
    # dropout_rate: tỷ lệ dropout để tránh overfitting
    def __init__(self, lpr_max_len, phase, class_num, dropout_rate):
        super(LPRNet, self).__init__()
        self.phase = phase
        self.lpr_max_len = lpr_max_len
        self.class_num = class_num

        # Backbone Network
        self.backbone = nn.Sequential(
            # Layer đầu tiên
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1), # RGB -> 64 channels
            nn.BatchNorm2d(num_features=64),  # Chuẩn hóa
            nn.ReLU(),  # Activation
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 1, 1)),  # Pooling
            # Block thứ nhát (i = 4..7)
            small_basic_block(ch_in=64, ch_out=128),  # 64 -> 128 channels
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(2, 1, 2)),  # Giảm kích thước
            # Block thứ hai (i = 8..11)
            small_basic_block(ch_in=64, ch_out=256),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            # Block thứ ba (i = 12..15)
            small_basic_block(ch_in=256, ch_out=256), # 256 -> 256 channels
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(4, 1, 2)), # Giảm mạnh kích thước
            # Các layer cuối (i = 16..22)
            nn.Dropout(dropout_rate),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(1, 4), stride=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Conv2d(in_channels=256, out_channels=class_num, kernel_size=(13, 1), stride=1),  # Số lượng class
            nn.BatchNorm2d(num_features=class_num),
            nn.ReLU(),
        )
        # Container Network: Nhận input có 448 + class_num channels, xuất ra class_num channels, sử dụng conv 1x1 để kết hợp thông tin
        self.container = nn.Sequential(
            nn.Conv2d(in_channels=448+self.class_num, out_channels=self.class_num, kernel_size=(1, 1), stride=(1, 1)),
        )

    # Forward Pass: lưu trữ feature maps
    def forward(self, x):
        keep_features = list()
        for i, layer in enumerate(self.backbone.children()):
            x = layer(x)
            if i in [2, 6, 13, 22]:
                keep_features.append(x)

        # Global context processing: Kết hợp thông tin từ nhiều tầng khác nhau, sử dụng thông tin toàn cục để cải thiện nhận dạng, chuẩn hóa feature để ổn định training
        global_context = list()
        for i, f in enumerate(keep_features):
            if i in [0, 1]:
                f = nn.AvgPool2d(kernel_size=5, stride=5)(f)
            if i in [2]:
                f = nn.AvgPool2d(kernel_size=(4, 10), stride=(4, 2))(f)
            f_pow = torch.pow(f, 2) # bình phương từng element
            f_mean = torch.mean(f_pow)  # tính mean của bình phương
            f = torch.div(f, f_mean)  # chuẩn hóa
            global_context.append(f)

        # Kết hợp và xuất kết quả
        x = torch.cat(global_context, 1)  # nối các feature maps
        x = self.container(x) # qua container network
        logits = torch.mean(x, dim=2) # average pooling theo chiều cao

        return logits

# Khởi tạo mô hình
# def build_lprnet(lpr_max_len=8, phase=False, class_num=66, dropout_rate=0.5):
def build_lprnet(lpr_max_len=8, phase=False, class_num=37, dropout_rate=0.5):
    Net = LPRNet(lpr_max_len, phase, class_num, dropout_rate)
    return Net.train() if phase == "train" else Net.eval()
