import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np

os.makedirs('graphs', exist_ok=True)

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

# Distribuição por sexo
print("---- Distribuição por sexo")
print(df.groupby('sexo')['encargos'].describe())

# Distribuição por fumante
print("---- Distribuição por fumante")
print(df.groupby('fumante')['encargos'].describe())

#Distribuição por sexo / fumante
print("---- Distribuição por sexo / fumante")
print(df.groupby(['sexo', 'fumante'])['encargos'].describe())

#Distribuição por sexo / regiao
print("---- Distribuição por sexo / regiao")
print(df.groupby(['sexo', 'regiao'])['encargos'].describe())

#Distribuição por idade / regiao
print("---- Distribuição por idade / regiao")
print(df.groupby(['idade', 'regiao'])['encargos'].describe())


#medias e contagens para todas as colunas
print(df.groupby('regiao').agg(
    media_encargos=('encargos', 'mean'),
    mediana_encargos=('encargos', 'median'),
    contagem=('encargos', 'count')
))
print(df.groupby('fumante').agg(
    media_imc=('imc', 'mean'),
    media_encargos=('encargos', 'mean'),
    contagem=('encargos', 'count')
))
print(df.groupby('sexo').agg(
    media_imc=('imc', 'mean'),
    media_encargos=('encargos', 'mean'),
    contagem=('encargos', 'count')
))

# Percebe-se até aqui que os encargos são maiores para os fumantes e para as mulheres
# E que as regiões sul e sudeste apresentam maiores encargos

#Geração de gráficos para melhor visualização dos dados
# Encargos por região
plt.figure(figsize=(10, 6))
sns.barplot(x='regiao', y='encargos', hue='regiao', data=df) #barras
plt.title('Encargos por Região')
plt.xlabel('Região')
plt.ylabel('Encargos')
plt.savefig('graphs/encargos_por_regiao.png', bbox_inches='tight')
plt.close()

# Encargos por sexo
plt.figure(figsize=(10, 6))
sns.barplot(x='sexo', y='encargos', hue='sexo', data=df)
plt.title('Encargos por Sexo')
plt.xlabel('Sexo')
plt.ylabel('Encargos')
plt.savefig('graphs/encargos_por_sexo.png', bbox_inches='tight')
plt.close()

# Encargos por fumante
plt.figure(figsize=(10, 6))
sns.barplot(x='fumante', y='encargos', hue='fumante', data=df)
plt.title('Encargos por Fumante')
plt.xlabel('Fumante')
plt.ylabel('Encargos')
plt.savefig('graphs/encargos_por_fumante.png', bbox_inches='tight')
plt.close()

#Pré processamento dos dados
print("---- Pré processamento dos dados")

#padroniza os valores para fumante e sexo e idade
df['fumante'] = df['fumante'].str.strip().str.capitalize()
df['sexo'] = df['sexo'].str.strip().str.capitalize()
df['idade'] = pd.to_numeric(df['idade'], errors='coerce')

# Remove os dados inválidos
df = df.dropna()
df = df[df['sexo'].isin(['Masculino', 'Feminino'])]
df = df[df['fumante'].isin(['Sim', 'Não'])]
df = df[(df['idade'] >= 0) & (df['idade'] <= 130)]

print(df.head())

# Converta variáveis categóricas em formatos numéricos adequados para modelagem
print("---- Converta variáveis categóricas em formatos numéricos adequados para modelagem")
df_encoded = pd.get_dummies(df, columns=['sexo', 'fumante', 'regiao'], drop_first=True) #prevenindo multicolinearidade

X = df_encoded.drop('encargos', axis=1)
y = df_encoded['encargos']











