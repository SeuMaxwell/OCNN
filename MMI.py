import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from optical_mmi import OpticalMMIConv2d

# ---------------------------
# 1. 数据准备
# ---------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_loader = DataLoader(
    datasets.MNIST('./data', train=True, download=True, transform=transform),
    batch_size=64, shuffle=True)
test_loader = DataLoader(
    datasets.MNIST('./data', train=False, transform=transform),
    batch_size=1000, shuffle=False)

# ---------------------------
# 2. 定义使用 MMI 的 CNN
# ---------------------------
class MMIMNIST(nn.Module):
    def __init__(self):
        super().__init__()
        # 用 OpticalMMIConv2d 替代 nn.Conv2d
        self.conv1 = OpticalMMIConv2d(in_channels=1, out_channels=32, kernel_size=3)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = OpticalMMIConv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))   # 光学卷积 + ReLU
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return F.log_softmax(self.fc2(x), dim=1)

# ---------------------------
# 3. 训练与验证流程
# ---------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MMIMNIST().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

def train_epoch():
    model.train()
    total_loss = 0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f'Train Loss: {total_loss/len(train_loader):.4f}')

def evaluate():
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
    acc = correct / len(test_loader.dataset)
    print(f'Test Accuracy: {acc*100:.2f}%')
    return acc

# ---------------------------
# 4. 运行
# ---------------------------
for epoch in range(1, 6):
    print(f'\nEpoch {epoch}')
    train_epoch()
    evaluate()
