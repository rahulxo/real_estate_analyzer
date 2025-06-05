import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error


df = pd.read_csv("train.csv")
#print(df.columns)


numeric_features = ["GrLivArea", "TotalBsmtSF", "GarageArea", "YearBuilt"]
categorical_features = ["Neighborhood", "BldgType"]
target = "SalePrice"

df = df[numeric_features + categorical_features + [target]].dropna()


numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown="ignore")

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)


model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])


X = df[numeric_features + categorical_features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
print(f"MAE: ${mae:,.2f}")


scores = cross_val_score(model, X, y, cv=5, scoring="neg_mean_absolute_error")
mean_mae = -scores.mean()
print(f"Cross-validated MAE: ${mean_mae:,.2f}")
