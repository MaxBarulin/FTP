import torch
import pandas as pd # Добавляем импорт pandas

def predict(model, scaler, new_data):
    # Ожидается, что new_data - это список списков, например: [[D, d, H]]
    # Создаем DataFrame с названиями колонок
    new_data_df = pd.DataFrame(new_data, columns=['D', 'd', 'H'])
    # Преобразуем данные с помощью scaler
    new_data_scaled = scaler.transform(new_data_df)
    new_X = torch.tensor(new_data_scaled, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        predicted_norm_time = model(new_X)
    return predicted_norm_time.item()