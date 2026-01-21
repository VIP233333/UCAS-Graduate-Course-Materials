import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch


# 转换为张量
transform = transforms.ToTensor()

# 下载训练集和测试集
train_data = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_data = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# 构建数据加载器
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=1000, shuffle=False)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 建立第一卷积层(Conv2d) -> 激励函数(ReLU) -> 池化(MaxPooling)
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        # 建立第二卷积层(Conv2d) -> 激励函数(ReLU) -> 池化(MaxPooling)
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # 建立第三卷积层(Conv2d)
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,  # 输入通道数与第二层输出一致
                out_channels=64,  # 增加特征图数量
                kernel_size=3,    # 使用较小的卷积核
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(64),  # 批归一化，加速收敛并提高稳定性
            nn.ReLU(),
            nn.MaxPool2d(2)  # 输出大小: (64, 3, 3)
        )
        
        # 调整全连接层输入尺寸
        self.dropout = nn.Dropout(0.5)  # 使用Dropout层防止过拟合
        self.out = nn.Linear(64 * 3 * 3, 10)  # 根据新的特征图尺寸调整

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)  # 通过第三层卷积
        x = x.view(x.size(0), -1)
        x = self.dropout(x)  # 在全连接前加入dropout
        output = self.out(x)
        return output

# 使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 增加训练周期
num_epochs = 10  # 10轮

# 学习率调度器
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# 开始训练
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # 每个epoch后更新学习率
    scheduler.step()
    
    # 打印每个epoch的平均损失
    avg_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Average Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")

model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")