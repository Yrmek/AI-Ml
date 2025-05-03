import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# 1. Загрузка сохраненной модели
model = load_model("my_model.keras")  # Укажите путь к вашей модели


# 2. Загрузка и предобработка изображения
def predict_single_image(img_path, target_size=(64, 64)):
    # Загрузка изображения
    img = image.load_img(img_path, target_size=target_size)

    # Преобразование в массив и нормализация
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Добавление батч-размерности

    # 3. Предсказание
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction[0])
    confidence = np.max(prediction[0])

    # 4. Визуализация
    plt.imshow(img)
    plt.title(f"Class: {predicted_class}, Confidence: {confidence:.2f}")
    plt.axis("off")
    plt.show()

    return predicted_class, confidence


# 5. Пример использования
img_path = "test_y.jpg"  # Укажите путь к тестовому изображению
predicted_class, confidence = predict_single_image(img_path)

print(f"Предсказанный класс: {predicted_class}")
print(f"Уверенность модели: {confidence:.2%}")