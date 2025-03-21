import pandas as pd

df = pd.read_csv("tested.csv")

df.head()

df["Age"] = df["Age"].fillna(df["Age"].median())

df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

from sklearn.preprocessing import MinMaxScaler, StandardScaler

scaler = MinMaxScaler()  # или StandardScaler()
df["Age"] = scaler.fit_transform(df[["Age"]])

df = pd.get_dummies(df, columns=["Embarked"], drop_first=True)
df = pd.get_dummies(df, columns=["Sex"], drop_first=True)
df.to_csv("processed_titanic.csv", index=False)