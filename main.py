import pandas as pd

df = pd.read_csv('database/insurance_brasil_simulado.csv')

print(df.head())

df.head()
df.describe()
# df.info()

print(df.describe());
# print(df.info());

for column in ['sexo', 'fumante', 'regiao']:
    print(df[column].value_counts())
    print(f"Valores únicos da coluna {column}: {df[column].unique()}")


