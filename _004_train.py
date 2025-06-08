import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from _002_MyDataset import MyDataSet
from _001_split_data import get_train_and_val
from _003_model import AlexNet
import time

t1 = time.time()

# 1. 读取数据集路径和标签
train_data = r'train.txt'
val_data = r'val.txt'
train_img_path, train_labels, val_img_path, val_labels, classes = get_train_and_val(train_data, val_data)

# 2. 数据预处理
data_transform = {
    "train": transforms.Compose([
        transforms.RandomResizedCrop(227),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    "val": transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}

# 3. 实例化数据集
train_dataset = MyDataSet(images_path=train_img_path,
                          images_class=train_labels,
                          transform=data_transform["train"])
val_dataset = MyDataSet(images_path=val_img_path,
                        images_class=val_labels,
                        transform=data_transform["val"])

# 4. 创建数据加载器
batch_size = 8
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           pin_memory=True)
val_loader = torch.utils.data.DataLoader(val_dataset,
                                         batch_size=batch_size,
                                         shuffle=False,
                                         pin_memory=True)

t2 = time.time()
print(f"训练集图像: {len(train_dataset)}, 验证集图像: {len(val_dataset)}")
print(f'数据加载耗时: {round((t2 - t1), 4)}s')

# 5. 初始化模型
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = AlexNet(num_classes=4, init_weights=True).to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0002 )

# 6. 训练循环
epochs = 50
best_acc = 0.0
for epoch in range(epochs):
    # 训练阶段
    model.train()
    train_loss, train_correct, total = 0.0, 0, 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

    # 验证阶段
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            val_loss += loss_function(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    # 打印epoch结果
    train_acc = 100 * train_correct / total
    val_acc = 100 * val_correct / val_total
    print(f'epoch {epoch+1}/{epochs} '
          f'train_loss: {train_loss/len(train_loader):.3f} '
          f'val_acc: {train_acc:.2f}% '
          f'val_loss: {val_loss/len(val_loader):.3f} '
          f'train_acc: {val_acc:.2f}%')

    # 保存最佳模型
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), 'AlexNet.pth')

print('训练完成')