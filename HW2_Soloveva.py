import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

torch.manual_seed(1000)


class TitanicDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.df = pd.read_csv('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv')
        # в данном датасете пустые значения имеют
        # колонки 'Age','Cabin' 'Embarked'
        # возрасты являются важной информацией, из значения
        # можно заполнить медианой
        self.df['Age'] = self.df['Age'].fillna(self.df['Age'].median())
        # пустые категориальные значения можно заполнить
        # значением Unknown
        self.df['Cabin'] = self.df['Cabin'].fillna('Unknown')
        self.df['Embarked'] = self.df['Embarked'].fillna('Unknown')
        # (хотя наиболее простым выходом относительно колонки
        # 'Cabin' было бы просто выбросить ее из датасета,
        # поскольку она не содержит важной информации для поиска
        # целевой переменной, при этом имеет много пустых значений
        # self.df = self.df.drop(['Cabin'], axis=1)
        # поскольку переменная 'Sex' имеет лишь 2 уникальных значения
        # ее можно закодировать в одну колонку нулями и единицами
        self.df['Sex'] = self.df['Sex'].replace({'male': 1, 'female': 0})

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, item):
        row = self.df.iloc[item]
        alive = torch.Tensor([1, 0])
        dead = torch.Tensor([0, 1])
        y = alive if [row['Survived']] else dead
        x = torch.Tensor([row['Age'], row['Sex'], row['Fare'], row['SibSp'], row['Pclass']])
        return x, y

titanic_dataset = TitanicDataset()
dataloader = DataLoader(dataset=titanic_dataset, batch_size=1, shuffle=True)

loss_fn = torch.nn.CrossEntropyLoss()

class SurvivalPredictorPerciptrone(torch.nn.Module):

  def __init__(self, input_size: int, hidden_size: int, output_size: int):
    super().__init__()
    self.fully_connected_layer = torch.nn.Linear(input_size, hidden_size)
    self.out_layer = torch.nn.Linear(hidden_size, output_size, bias=True)
    self.sigmoid = torch.nn.Sigmoid()
    self.tanh = torch.nn.Tanh()
    self.silu = torch.nn.SiLU()
    self.lrelu = torch.nn.LeakyReLU()
    self.relu = torch.nn.ReLU()
    self.softmax = torch.nn.Softmax()


  def forward(self, x):
    x = self.fully_connected_layer(x)
    x = self.lrelu(x)
    x = self.out_layer(x)
    x = self.softmax(x)
    return x

model = SurvivalPredictorPerciptrone(5, 200, 2)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)

num_epoch = 20

for epoch in range(num_epoch):
  for x, y in titanic_dataset:
    optimizer.zero_grad()
    prediction = model(x)
    loss = loss_fn(prediction, y)
    print(loss)
    loss.backward()
    optimizer.step()

# результаты:
# 1. модель с сигмоидальной функцией активации показывает большие потери на первых
# этапах, но она быстро уменьшается
# 2. функции ReLU, SiLU, Tanh и LeakyReLU  дают выдают похожие значения и на первых этапах
# показывают потерю меньшую, чем сигмоидальная, но дают близкие к ней результаты
# на последних эпохах
# 3. увеличение lr немного уменьшает значение потери на 20-й эпохе, уменьшение
# замедляет обучение и требует большего числа эпох
# 4. увеличение размера внутреннего слоя ускоряет обучение но не даёт
# принципиального уменьшения значения потери
# 5. кодирование пола с помощью one-hot encoding принципиальной разницы также не дало