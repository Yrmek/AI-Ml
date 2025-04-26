import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt


# Загружаем датасет titanic
df = pd.read_csv("processed_titanic.csv")

X = df.drop(columns=['Embarked_Q','Ticket','Name', 'Cabin','SibSp','Fare'])
y = df['Embarked_Q']


# Разделяем данные на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# Случайный лес
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)


# Градиентный бустинг
gb = GradientBoostingClassifier(random_state=42)
gb.fit(X_train, y_train)
gb_pred = gb.predict(X_test)


# Оценка результатов
print("Random Forest Accuracy:", accuracy_score(y_test, rf_pred))
print("Gradient Boosting Accuracy:", accuracy_score(y_test, gb_pred))
print("Random Forest F1-Score:", f1_score(y_test, rf_pred, average='weighted'))
print("Gradient Boosting F1-Score:", f1_score(y_test, gb_pred, average='weighted'))


# Перекрестная проверка
rf_cv = cross_val_score(rf, X, y, cv=5, scoring='accuracy')
gb_cv = cross_val_score(gb, X, y, cv=5, scoring='accuracy')


print("Random Forest Cross-validation accuracy:", rf_cv.mean())
print("Gradient Boosting Cross-validation accuracy:", gb_cv.mean())


# Визуализация результатов
models = ['Random Forest', 'Gradient Boosting']
accuracy = [accuracy_score(y_test, rf_pred), accuracy_score(y_test, gb_pred)]


plt.bar(models, accuracy)
plt.title("Model Comparison")
plt.ylabel("Accuracy")
plt.show()
