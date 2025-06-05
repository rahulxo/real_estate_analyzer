import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("train.csv")

print(df.head())

print(df.columns)
missing = df.isna().sum()
print(missing[missing > 0])
df['LotFrontage'].median()
df.dropna()



sns.histplot(df['SalePrice'], bins=40)
plt.show()