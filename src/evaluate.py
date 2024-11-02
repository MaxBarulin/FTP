import torch
from torch.utils.data import DataLoader
from torch import nn

def evaluate_model(model, test_dataset):
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    criterion = nn.MSELoss()
    model.eval()
    losses = []

    with torch.no_grad():
        for batch_X, batch_y in test_loader: # Добавлено "in test_loader"
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            losses.append(loss.item())

    avg_loss = sum(losses) / len(losses)
    print(f'Средняя ошибка на тестовой выборке: {avg_loss:.4f}')
    return avg_loss