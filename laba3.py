import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, confusion_matrix
import matplotlib.pyplot as plt

# Загрузка данных
df = pd.read_csv("processed_titanic.csv")

X = df.drop(columns=['Embarked_S','Ticket','Name', 'Cabin','SibSp','Fare'])
y = df['Embarked_S']

# Разделяем данные на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создаем и обучаем модель дерева решений
tree_model = DecisionTreeClassifier(random_state=42)  # Ограничиваем глубину для лучшей визуализации
tree_model.fit(X_train, y_train)

# Предсказания на тестовых данных
y_pred = tree_model.predict(X_test)

# Оценка модели
precision = precision_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("Дерево решений:")
print("Точность классификации (Precision):", precision)
print("\nМатрица ошибок:")
print(conf_matrix)

# Визуализация дерева решений
plt.figure(figsize=(20, 10))
plot_tree(tree_model,
          feature_names=X.columns,
          class_names=['Not Survived', 'Survived'],
          filled=True,
 proportion=True
          )
plt.title("Дерево решений для классификации пассажиров")
plt.show()