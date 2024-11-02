from src.data_processing import load_data, preprocess_data
from src.dataset import FlangeDataset
from src.model import SimpleRegressor
from src.train import train_model
from src.evaluate import evaluate_model
from src.predict import predict
from sklearn.model_selection import KFold

def main():
    # Загрузка и предобработка данных
    data = load_data('data/data.csv')
    features_scaled, targets, scaler = preprocess_data(data)

    # Создание набора данных
    dataset = FlangeDataset(features_scaled, targets)

    # Кросс-валидация
    kf = KFold(n_splits=5)
    losses = []
    for fold, (train_index, test_index) in enumerate(kf.split(features_scaled)):
        print(f'Fold {fold+1}')
        X_train, X_test = features_scaled[train_index], features_scaled[test_index]
        y_train, y_test = targets[train_index], targets[test_index]

        train_dataset = FlangeDataset(X_train, y_train)
        test_dataset = FlangeDataset(X_test, y_test)

        # Инициализация модели
        model = SimpleRegressor()

        # Обучение модели
        model = train_model(model, train_dataset, num_epochs=500)

        # Оценка модели
        avg_loss = evaluate_model(model, test_dataset)
        losses.append(avg_loss)

    print(f'Средняя ошибка по всем фолдам: {sum(losses)/len(losses):.4f}')

    # Предсказание на новых данных
    new_flange = [[600, 400, 38]] # Замените на реальные значения
    predicted_time = predict(model, scaler, new_flange)
    print(f'Предсказанная норма времени для фланца {new_flange}: {predicted_time:.2f}')

if __name__ == '__main__':
    main()
