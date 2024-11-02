import torch
from torch.utils.data import DataLoader
from torch import nn

def train_model(model, train_dataset, num_epochs=1000, batch_size=4, learning_rate=0.01):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        for batch_X, batch_y in train_loader: # Добавлено "in train_loader"
            # Прямой проход
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)

            # Обратное распространение и оптимизация
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    return model