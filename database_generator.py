import pandas as pd
import random
import numpy as np

random.seed(42)
np.random.seed(42)

sex_options = ['Masculino', 'Feminino']
smoker_options = ['Sim', 'Não']
region_options = ['Sudeste', 'Sul', 'Nordeste', 'Centro-Oeste', 'Norte']


n = 10000
data = {
    'age': np.random.randint(18, 66, size=n),
    'sex': np.random.choice(sex_options, size=n),
    'bmi': np.round(np.random.normal(loc=27, scale=5, size=n), 1),
    'children': np.random.randint(0, 5, size=n),
    'smoker': np.random.choice(smoker_options, size=n, p=[0.15, 0.85]),
    'region': np.random.choice(region_options, size=n)
}

# Criar DataFrame
df = pd.DataFrame(data)

# Gerar encargos
base_charge = 500 + (df['age'] * 20) + (df['bmi'] * 30) + (df['children'] * 100)

# Classificar fumantes
base_charge += np.where(df['smoker'] == 'Sim', 5000, 0)

# Variação por região
region_adjustment = {
    'Sudeste': 1.2,
    'Sul': 1.1,
    'Nordeste': 0.9,
    'Centro-Oeste': 1.0,
    'Norte': 0.8
}

df['charges'] = (base_charge * df['region'].map(region_adjustment)).round(2)

print(df.head());

csv_path = 'database/insurance_brasil_simulado.csv'

df.to_csv(csv_path, index=False)

print("Arquivo 'insurance_brasil_simulado.csv' gerado com sucesso!")
