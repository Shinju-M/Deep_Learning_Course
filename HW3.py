import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

torch.manual_seed(1000)

# загрузка полного ДС
cifar_dataset = datasets.CIFAR100(download=True, root='/content/sample_data')
#print(len(cifar_dataset))
# датасет содержит 50000 картинок

# таргет содержит 100 классов
#print(len(cifar_dataset.classes))
# print(cifar_dataset.class_to_idx)

# индексы для обучающей и тестовой выборок
train_indices = torch.arange(15000)
test_indices = torch.arange(4500)

# создание тренировочной выборки
train_cifar_dataset = datasets.CIFAR100(download=True, root='./', transform=
    transforms.ToTensor(), train=True
)
train_cifar_dataset = Subset(train_cifar_dataset, train_indices)
train_cifar_dataloader = DataLoader(dataset=train_cifar_dataset, batch_size=1)

# создание тестовой выборки
test_cifar_dataset = datasets.CIFAR100(download=True, root='./', transform=
    transforms.ToTensor(), train=False
)
test_cifar_dataset = Subset(test_cifar_dataset, test_indices)
test_cifar_dataloader = DataLoader(dataset=test_cifar_dataset, batch_size=1)

loss_fn = torch.nn.BCELoss()


class CIFARPredictorPerciptrone(torch.nn.Module):

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.fully_connected_layer = torch.nn.Linear(input_size, hidden_size)
        self.out_layer = torch.nn.Linear(hidden_size, output_size, bias=True)
        self.sigmoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()
        self.silu = torch.nn.SiLU()
        self.lrelu = torch.nn.LeakyReLU()
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)
        self.dropout = torch.nn.Dropout(p=0.1)

    def forward(self, x):
        x = torch.flatten(x, 1) # flatten превращает матрицу в вектор
        x = self.fully_connected_layer(x)
        x = self.relu(x)
        #x = self.dropout(x)
        x = self.out_layer(x)
        x = self.softmax(x)
        return x


model = CIFARPredictorPerciptrone(3072, 200, 100)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

writer = SummaryWriter(log_dir='./runs')

num_epoch = 30

for epoch in range(num_epoch):
    error = 0
    for x, y in train_cifar_dataloader:
        model.train()
        optimizer.zero_grad()
        prediction = model(x)
        zero_tensor = torch.zeros_like(prediction)
        zero_tensor[:, y[0]] = 1
        y = zero_tensor
        loss = loss_fn(prediction, y)
        error += loss
        # print(loss)
        loss.backward()
        optimizer.step()

    print(error / len(train_cifar_dataset))
    writer.add_scalar('Loss', error / len(train_cifar_dataset), epoch)

    correct_guess = 0
    for x, y in test_cifar_dataloader:
        model.eval()
        prediction = model(x)
        predicted_indices = torch.argmax(prediction)
        correct_guess += (predicted_indices == y).float().sum()

    print(f'Accuracy {correct_guess / len(test_cifar_dataset)}')
    writer.add_scalar('Accuracy', correct_guess / len(test_cifar_dataset), epoch)

# Выводы:
# 1. Уменьшение learning rate ускоряет обучение сети, но accuracy уменьшается
# или затормаживается к 20-й эпохе (ошибка продолжает уменьшаться),
# что может указывать на переобучение (добавление слоя dropout позволяет это предотвратить
# даже с большим числом эпох, однако замедляет обучение)
# 2. Увеличение выборки улучшает результаты обучения модели, при этом значение
# accuracy увеличиватеся даже при увеличении числа эпох (до 30) при добавлении dropout
# (т.е. модель не переобучается)
# 3. Увеличение или уменьшение числа нейронов в скрытом слое
# приводит к ухудшению результатов обучения
# 4. tensorboard рисует красивые графики