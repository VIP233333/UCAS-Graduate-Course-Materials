import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from tqdm import tqdm
import matplotlib

matplotlib.use('TkAgg')  # 或者 'Qt5Agg'
import matplotlib.pyplot as plt
import numpy as np
import os

# 加载和预处理数据
trans_train = transforms.Compose(
    [transforms.RandomCrop(32, padding=4),  # 将给定图像随机裁剪为不同的大小和宽高比，然后缩放所裁得到的图像为指定大小
     transforms.RandomHorizontalFlip(),  # 以给定的概率随机水平旋转给定的PIL图像，
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet统计值
                          std=[0.229, 0.224, 0.225])])

trans_vaild = transforms.Compose(
    [
        transforms.ToTensor(),  # 将PIL Image或者ndarray 转换为tensor,并归一化至[0,1]。
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])  # 先减均值，除标准差
# 对训练集做数据增强，增加模型的鲁棒性，而对测试集不做翻转和增强

trainset = torchvision.datasets.CIFAR10(root="./cifar10", train=True, download=True, transform=trans_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True, num_workers=4, pin_memory=True)
testset = torchvision.datasets.CIFAR10(root='./cifar10', train=False, download=False, transform=trans_vaild)
testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# attention 结构
class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim), nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # 生成Q、K、V
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)  # Softmax归一化

        out = torch.matmul(attn, v)  # 注意力加权
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),  # 扩展维度
            nn.GELU(),  # 高斯误差线性单元
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),  # 恢复维度
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([
            nn.ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                FeedForward(dim, mlp_dim, dropout=dropout),
            ])
            for _ in range(depth)
        ])

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim,
                 pool='cls', channels=3, dim_head=64, dropout=0.1, emb_dropout=0.1):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        assert image_height % patch_height == 0 and image_width % patch_width == 0

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))  # 64 patches + CLS
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.pool = pool
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, img):
        x = self.to_patch_embedding(img)  # 分块嵌入 [b, 64, 256]
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)  # 添加CLS令牌 [b, 65, 256]
        x += self.pos_embedding[:, :(n + 1)]  # 位置编码
        x = self.dropout(x)  # 8层Transformer编码
        x = self.transformer(x)  # 提取CLS令牌输出
        x = x.mean(1) if self.pool == 'mean' else x[:, 0]
        return self.mlp_head(x)  # 分类头输出10维


# 训练

device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
# device = torch.device("cuda:1")  # 0独显，1为核显
print("Selected device:", device)
print("Device name:", torch.cuda.get_device_name(device.index))
net = ViT(
    image_size=32,
    patch_size=4,
    num_classes=10,
    dim=256,
    depth=8,
    heads=8,
    mlp_dim=512,
    dropout=0.1,
    emb_dropout=0.1,
).to(device)

torch.backends.cudnn.benchmark = True
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=0.05)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

best_acc = 0

plt.close('all')
plt.ion()
fig, ax = plt.subplots()
line, = ax.plot([], [], lw=2)
ax.set_xlabel('batch_num')
ax.set_ylabel('acc')
ax.set_title('Training Accuracy')
ax.set_ylim(0, 1)
ax.grid(True)
plt.show(block=False)
batch_acc_list = []


def train(epoch):
    net.train()
    running_loss, correct, total = 0., 0, 0
    for inputs, targets in tqdm(trainloader, desc=f'Train {epoch}'):
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        acc = correct / total
        batch_acc_list.append(acc)
        line.set_data(range(len(batch_acc_list)), batch_acc_list)
        ax.set_xlim(0, len(batch_acc_list))
        ax.relim();
        ax.autoscale_view()
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        plt.pause(0.001)

    epoch_acc = 100. * correct / total
    print(f'Epoch {epoch} | Train Loss: {running_loss / len(trainloader):.4f} | Train Acc: {epoch_acc:.2f}%')


early_stop_patience = 15
epochs_no_improve = 0
best_acc = 0
early_stop = False

def evaluate(epoch):
    global best_acc, epochs_no_improve, early_stop
    net.eval()
    test_loss, correct, total = 0., 0, 0
    with torch.no_grad():
        for inputs, targets in tqdm(testloader, desc=f'Val {epoch}'):
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    acc = 100. * correct / total
    scheduler.step()
    print(f'→ Val Loss: {test_loss / len(testloader):.4f} | Val Acc: {acc:.2f}%')

    # 早停
    if acc > best_acc:
        best_acc = acc
        epochs_no_improve = 0
        os.makedirs('checkpoint', exist_ok=True)
        torch.save({'net': net.state_dict(), 'acc': acc, 'epoch': epoch},
                   f'checkpoint/vit_patch4_best.pth')  # 保存最佳模型
        print('  ✔ Best model saved.')
    else:
        epochs_no_improve += 1
        print(f'  No improvement for {epochs_no_improve} epochs')
        
    if epochs_no_improve >= early_stop_patience:  # 早停判断
        early_stop = True
        
    return early_stop


if __name__ == '__main__':
    for epoch in range(1, 101):
        train(epoch)
        should_stop = evaluate(epoch)
        if should_stop:
            print(f"Early stopping triggered at epoch {epoch}")
            break

    plt.ioff()
    plt.savefig('train_accuracy_curve.png', dpi=300)
    plt.show()
