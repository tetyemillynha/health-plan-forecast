import pandas as pd

# Carregue a base de dados e explore suas características;
df = pd.read_csv('database/insurance_brasil_simulado.csv')
print(df.head())


# Analise estatísticas descritivas e visualize distribuições relevantes.
df.head()
df.describe()
# print(df.info());
# print(df.describe());

# Analisando os valores nulos de cada coluna (numericas)
print(df.isnull().sum());

# Analisando os valores únicos de cada coluna (categoricas)
for column in ['sexo', 'fumante', 'regiao']:
    # print(df[column].value_counts())
    print(f"Valores únicos da coluna {column}: {df[column].unique()}")

# Analisando os valores nulos de cada coluna (numericas)
df.isnull().sum()


# Vizualização de distribuições relevantes.

# Distribuição por região
print("---- Distribuição por região")
print(df.groupby('regiao')['encargos'].describe())

# Distribuição por idade
print("---- Distribuição por idade")
print(df.groupby('idade')['encargos'].describe())

# Distribuição por sexo
print("---- Distribuição por sexo")
print(df.groupby('sexo')['encargos'].describe())

# Distribuição por fumante
print("---- Distribuição por fumante")
print(df.groupby('fumante')['encargos'].describe())

#Distribuição por sexo / fumante
print("---- Distribuição por sexo / fumante")
print(df.groupby(['sexo', 'fumante'])['encargos'].describe())

#Distribuição por sexo / idade
print("---- Distribuição por sexo / idade")
print(df.groupby(['sexo', 'idade'])['encargos'].describe())

#Distribuição por idade / fumante
print("---- Distribuição por idade / fumante")
print(df.groupby(['idade', 'fumante'])['encargos'].describe())

#Distribuição por sexo / regiao
print("---- Distribuição por sexo / regiao")
print(df.groupby(['sexo', 'regiao'])['encargos'].describe())

#Distribuição por idade / regiao
print("---- Distribuição por idade / regiao")
print(df.groupby(['idade', 'regiao'])['encargos'].describe())



# Boxplot por região
