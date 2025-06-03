import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score

os.makedirs('graphs', exist_ok=True)

language = "en"

if language == "en":
    file_name = 'insurance_international'
    label_sex_male = "Male"
    label_sex_female = "Female"
    label_smoker_yes = "Yes"
    label_smoker_no = "No"
else:
    file_name = 'insurance_brasil_simulado'
    label_sex_male = "Masculino"
    label_sex_female = "Feminino"
    label_smoker_yes = "Sim"
    label_smoker_no = "Não"

df = pd.read_csv(f'database/{file_name}.csv')
print(df.head())


# Analise estatísticas descritivas e visualize distribuições relevantes.
df.head()
df.describe()

# Analisando os valores nulos de cada coluna (numericas)
print(df.isnull().sum());

# Analisando os valores únicos de cada coluna (categoricas)
for column in ['sex', 'smoker', 'region']:
    # print(df[column].value_counts())
    print(f"Valores únicos da coluna {column}: {df[column].unique()}")

# Analisando os valores nulos de cada coluna (numericas)
df.isnull().sum()


# Vizualização de distribuições relevantes.
# Distribuição por região
print("---- Distribuição por região")
print(df.groupby('region')['charges'].describe())

# Distribuição por sexo
print("---- Distribuição por sexo")
print(df.groupby('sex')['charges'].describe())

# Distribuição por fumante
print("---- Distribuição por fumante")
print(df.groupby('smoker')['charges'].describe())

#Distribuição por sexo / fumante
print("---- Distribuição por sexo / fumante")
print(df.groupby(['sex', 'smoker'])['charges'].describe())

#Distribuição por sexo / regiao
print("---- Distribuição por sexo / regiao")
print(df.groupby(['sex', 'region'])['charges'].describe())

#Distribuição por idade / regiao
print("---- Distribuição por idade / regiao")
print(df.groupby(['age', 'region'])['charges'].describe())


#medias e contagens para todas as colunas
print(df.groupby('region').agg(
    media_charges=('charges', 'mean'),
    mediana_charges=('charges', 'median'),
    contagem=('charges', 'count')
))
print(df.groupby('smoker').agg(
    media_charges=('charges', 'mean'),
    mediana_charges=('charges', 'median'),
    contagem=('charges', 'count')
))
print(df.groupby('sex').agg(
    media_charges=('charges', 'mean'),
    mediana_charges=('charges', 'median'),
    contagem=('charges', 'count')
))

# Percebe-se até aqui que os encargos são maiores para os fumantes e para as mulheres
# E que as regiões sul e sudeste apresentam maiores encargos

#Geração de gráficos para melhor visualização dos dados
# Encargos por região
plt.figure(figsize=(10, 6))
sns.barplot(x='region', y='charges', hue='region', data=df) #barras
plt.title(f'Encargos por Região - {language}')
plt.xlabel('Região')
plt.ylabel('Encargos')
plt.savefig(f'graphs/encargos_por_regiao_{language}.png', bbox_inches='tight')
plt.close()

# Encargos por sexo
plt.figure(figsize=(10, 6))
sns.barplot(x='sex', y='charges', hue='sex', data=df)
plt.title(f'Encargos por Sexo - {language}')
plt.xlabel('Sexo')
plt.ylabel('Encargos')
plt.savefig(f'graphs/encargos_por_sexo_{language}.png', bbox_inches='tight')
plt.close()

# Encargos por fumante
plt.figure(figsize=(10, 6))
sns.barplot(x='smoker', y='charges', hue='smoker', data=df)
plt.title(f'Encargos por Fumante - {language}')
plt.xlabel('Fumante')
plt.ylabel('Encargos')
plt.savefig(f'graphs/encargos_por_fumante_{language}.png', bbox_inches='tight')
plt.close()

#Pré processamento dos dados
print("---- Pré processamento dos dados")
#padroniza os valores para fumante e sexo e idade
df['smoker'] = df['smoker'].str.strip().str.capitalize()
df['sex'] = df['sex'].str.strip().str.capitalize()
df['age'] = pd.to_numeric(df['age'], errors='coerce')

# Remove os dados inválidos
df = df.dropna()
df = df[df['sex'].isin([label_sex_male, label_sex_female])]
df = df[df['smoker'].isin([label_smoker_yes, label_smoker_no])]
df = df[(df['age'] >= 0) & (df['age'] <= 130)]
df = df[(df['bmi'] >= 10) & (df['bmi'] <= 60)]

print(df.head())

# Converta variáveis categóricas em formatos numéricos adequados para modelagem
print("---- Converta variáveis categóricas em formatos numéricos adequados para modelagem")
df_encoded = pd.get_dummies(df, columns=['sex', 'smoker', 'region'], drop_first=True) #prevenindo multicolinearidade

X = df_encoded.drop('charges', axis=1)
y = df_encoded['charges']

# Divide os dados em conjuntos de treinamento e teste (80 / 20) - sklearn
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Cria e treina o modelo de regressão linear
model_linearRegression = LinearRegression()
model_linearRegression.fit(X_train, y_train)
## Faz previsões
y_pred = model_linearRegression.predict(X_test)
# Avalia o modelo
mse_linearRegression = mean_squared_error(y_test, y_pred)
mae_linearRegression = mean_absolute_error(y_test, y_pred)
r2_linearRegression = r2_score(y_test, y_pred)

print("---- Avalia o modelo de Regressão Linear")
print(f"MSE: {mse_linearRegression}")
print(f"MAE: R$ {round(mae_linearRegression, 2)}")
print(f"R2: {r2_linearRegression}")

# Cria e treina o modelo de árvore de decisão
model_decisionTree = DecisionTreeRegressor()
model_decisionTree.fit(X_train, y_train)
## Faz previsões
y_pred_decisionTree = model_decisionTree.predict(X_test)
# Avalia o modelo
mse_decisionTree = mean_squared_error(y_test, y_pred_decisionTree)
mae_decisionTree = mean_absolute_error(y_test, y_pred_decisionTree)
r2_decisionTree = r2_score(y_test, y_pred_decisionTree)

print("---- Avalia o modelo de Árvore de Decisão")
print(f"MSE: {mse_decisionTree}")
print(f"MAE: R$ {round(mae_decisionTree, 2)}")
print(f"R2: {r2_decisionTree}")

# Cria e treina o modelo de Random Forest
model_randomForest = RandomForestRegressor()
model_randomForest.fit(X_train, y_train)
## Faz previsões
y_pred_randomForest = model_randomForest.predict(X_test)
# Avalia o modelo
mse_randomForest = mean_squared_error(y_test, y_pred_randomForest)
mae_randomForest = mean_absolute_error(y_test, y_pred_randomForest)
r2_randomForest = r2_score(y_test, y_pred_randomForest)

print("---- Avalia o modelo de Random Forest")
print(f"MSE: {mse_randomForest}")
print(f"MAE: R$ {round(mae_randomForest, 2)}")
print(f"R2: {r2_randomForest}")

# Faz validação cruzada para comparar os modelos
print("---- Faz validação cruzada para comparar os modelos")
kfold = KFold(n_splits=5, shuffle=True, random_state=42) #fazendo 5 folds

scores_linearRegression = cross_val_score(model_linearRegression, X, y, cv=kfold, scoring='r2')
scores_decisionTree = cross_val_score(model_decisionTree, X, y, cv=kfold, scoring='r2')
scores_randomForest = cross_val_score(model_randomForest, X, y, cv=kfold, scoring='r2')

print("---- Avaliação dos modelos")
print(f"R2 - Regressão Linear: {scores_linearRegression.mean()}")
print(f"R2 - Árvore de Decisão: {scores_decisionTree.mean()}")
print(f"R2 - Random Forest: {scores_randomForest.mean()}")

#monta tabela pra melhor visualização
modal_names = ['Regressão Linear', 'Árvore de Decisão', 'Random Forest']
r2_test = [r2_linearRegression, r2_decisionTree, r2_randomForest]
r2_test_std = [np.std(scores_linearRegression), np.std(scores_decisionTree), np.std(scores_randomForest)]
r2_test_mean = [np.mean(scores_linearRegression), np.mean(scores_decisionTree), np.mean(scores_randomForest)]

df_results = pd.DataFrame({
    'Modelo': modal_names,
    'R2 (teste unico)': [round(x, 4) for x in r2_test],
    'R2 (teste cruzado)': [round(x, 4) for x in r2_test_mean],
    'Desvio padrão': [round(x, 4) for x in r2_test_std]
})

print("---- Resultados dos modelos")
print(df_results.to_string(index=False));
df_results.to_csv(f'results/results_modelos_{language}.csv', index=False)


# Gráfico de barras - R²
model_names = ['Regressão Linear', 'Árvore de Decisão', 'Random Forest']
r2_teste = [r2_linearRegression, r2_decisionTree, r2_randomForest]
r2_cv = [np.mean(scores_linearRegression), np.mean(scores_decisionTree), np.mean(scores_randomForest)]

x = np.arange(len(model_names))
width = 0.35

plt.figure(figsize=(10, 6))
plt.bar(x - width/2, r2_teste, width, label='Teste')
plt.bar(x + width/2, r2_cv, width, label='Validação Cruzada')
# plt.ylim(0.95, 1.01)
plt.ylabel('R²')
plt.title(f'Comparação do R² dos Modelos - {language}')
plt.xticks(x, model_names)
plt.legend()
plt.tight_layout()
plt.savefig(f'graphs/r2_comparacao_modelos_{language}.png')
plt.close()


#  Gráfico real vs previsto
def plot_real_vs_previsto(y_test, y_pred, nome_modelo, filename):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Valor Real')
    plt.ylabel('Valor Previsto')
    plt.title(f'Real vs Previsto - {nome_modelo} - {language}')
    plt.tight_layout()
    plt.savefig(f'graphs/{filename}')
    plt.close()

# Aplica para cada modelo
plot_real_vs_previsto(y_test, y_pred, "Regressão Linear", f"real_vs_previsto_lr_{language}.png")
plot_real_vs_previsto(y_test, y_pred_decisionTree, "Árvore de Decisão", f"real_vs_previsto_dt_{language}.png")
plot_real_vs_previsto(y_test, y_pred_randomForest, "Random Forest", f"real_vs_previsto_rf_{language}.png")












