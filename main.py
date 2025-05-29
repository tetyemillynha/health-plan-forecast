import pandas as pd

df = pd.read_csv('database/insurance_brasil_simulado.csv')

print(df.head())

# Analisando os dados
df.head()
df.describe()
print(df.info());
print(df.describe());

# Analisando os valores nulos de cada coluna (numericas)
print(df.isnull().sum());

# Analisando os valores únicos de cada coluna (categoricas)
for column in ['sexo', 'fumante', 'regiao']:
    print(df[column].value_counts())
    print(f"Valores únicos da coluna {column}: {df[column].unique()}")

# Analisando os valores nulos de cada coluna (numericas)
df.isnull().sum()

