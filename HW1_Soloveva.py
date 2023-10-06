import torch

# в следующем датасете описана лапша быстрого приготовления
# со вкусом курицы нескольких производителей, оцененная
# по следующим критериям
# 1. солёность (0 - не солёная, 1 - очень солёная)
# 2. острота (0 - не острая, 1 - очень острая)
# 3. разнообразие приправы (0 - менее 2-х компонентов, 1 - 10 и более компонентов)
# 4. наличие мяса (0 - отсутствует, 1 - присутствует)
# все критерии, кроме последнего представлены числом от 0 до 1
# критерий "наличие мяса" представлен двумя значениями - 1 при наличии мяса
# и 0 при его отсутствии

rollton = torch.tensor([[1, 0.3, 1, 1]])
doshirak = torch.tensor([[0.4, 0.4, 0.8, 0]])
bigbon = torch.tensor([[0.6, 0.4, 0.8, 1]])
shin_ramyon = torch.tensor([[0.5, 1, 0.8, 0]])
samyang = torch.tensor([[0.7, 0.8, 0.9, 0]])


# целевая переменная - оценка вкусовых свойств лапши группой добровольцев
# (0 - очень низкие вкусовые свойства, 1 - очень высокие)
dataset = (
    (rollton, torch.tensor([[0.7]])),
    (doshirak, torch.tensor([[0.7]])),
    (bigbon, torch.tensor([[0.5]])),
    (shin_ramyon, torch.tensor([[0.9]])),
    (samyang, torch.tensor([[0.8]]))
)

torch.manual_seed(1000)

weights = torch.rand((1, 4), requires_grad=True)
bias = torch.rand((1,1), requires_grad=True)

loss_fn = torch.nn.MSELoss()

optimizer = torch.optim.SGD([weights, bias], lr=0.1)


def predict_taste_score(obj: torch.Tensor) -> torch.Tensor:
    return obj @ weights.T + bias


def calc_loss(pred_val: torch.Tensor, true_val: torch.Tensor) -> torch.Tensor:
    return loss_fn(pred_val, true_val)


num_epoch = 20


for i in range(num_epoch):
    for x, y in dataset:
        optimizer.zero_grad()
        score = predict_taste_score(x)
        loss = calc_loss(score, y)
        loss.backward()
        print(loss)
        optimizer.step()
        print(f'After update: {weights, bias}')
