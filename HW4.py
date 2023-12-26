import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader
from torch.utils.tensorboard import SummaryWriter

torch.manual_seed(1000)

writer = SummaryWriter()

train_indices = torch.arange(5000)
test_indices = torch.arange(1500)

# размер картинки в датасете - 32х32
train_cifar_dataset = Subset(datasets.CIFAR10(
    train=True,
    download=True, root='./',
    transform=transforms.ToTensor()),
    train_indices
)

test_cifar_dataset = Subset(datasets.CIFAR10(
    train=False, download=True, root='./', transform=transforms.ToTensor()),
    test_indices
)


train_dl = DataLoader(train_cifar_dataset, batch_size=1)
test_dl = DataLoader(test_cifar_dataset, batch_size=1)

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3, # устанавливаем канналов для цветных изображений
            out_channels=32,
            kernel_size=3,
            stride=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=1
        )

        self.conv3 = nn.Conv2d(
            in_channels=64,
            out_channels=126,
            kernel_size=3,
            stride=1
        )
        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.05)
        self.fc1 = nn.Linear(2304, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        # x = self.conv3(x)
        # x = F.relu(x)
        # x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout2(x)

        return F.softmax(x, dim=1)


model = ConvNet()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
loss_fn = torch.nn.CrossEntropyLoss()
num_epochs = 30


for epoch in range(num_epochs):
    error = 0
    for x, y in train_dl:
        model.train()
        optimizer.zero_grad()
        prediction = model(x)
        zero_tensor = torch.zeros_like(prediction)
        zero_tensor[:, y[0]] = 1
        y = zero_tensor
        loss = loss_fn(prediction, y)
        error += loss

        loss.backward()
        optimizer.step()

    writer.add_scalar('Loss', error/len(train_cifar_dataset), epoch)

    correct_guess = 0
    for x, y in test_dl:
        model.eval()
        prediction = model(x)
        predicted_indices = torch.argmax(prediction)
        correct_guess += (predicted_indices == y).float().sum()

    writer.add_scalar('Accuracy', correct_guess/len(test_cifar_dataset), epoch)


# 1. увеличение значения lr привело к увеличению скорости обучения, при этом accuracy
# продолжила расти при большем числе эпох (достигла значения около 0.5 на к 27-й эпохе)
# 2. добавление третьего свёрточного слоя при тех же значениях lr замедлило обучение
# но в целом accuracy продолжало расти к 30-й эпохе
# 3. увеличение размера матрицы замедлило обучение и привело к резким скачкам
# точности на разных эпохах при большом значении lr, при меньшем значении обучение
# проходит очень медленно
# 4. увеличение шага до 2-х в данном датасете привело к замедлению обучения при небольшом
# значении lr (оставалось одинаковым на протяжении 30-ти эпох) и резким скачкам accuracy
# при больших значениях lr (0.1)