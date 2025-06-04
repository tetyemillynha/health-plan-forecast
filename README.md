# Previsão de Encargos de Planos de Saúde

Este projeto tem como objetivo desenvolver modelos de *Machine Learning* para prever o valor dos encargos cobrados por planos de saúde, com base em características individuais dos beneficiários.

## Objetivo

Prever o valor dos encargos a partir das seguintes variáveis:

* Idade
* Sexo
* IMC (Índice de Massa Corporal)
* Número de filhos
* Fumante (sim/não)
* Região de residência
* Encargos cobrados pelo plano de saúde

## Modelos Utilizados

Foram implementados e avaliados três modelos preditivos:

* Regressão Linear
* Árvore de Decisão
* Random Forest

## Estrutura do Projeto

```
.
├── database/
│   └── insurance_brasil_simulado.csv   # Base de dados simulada
│   └── insurance_international.csv   # Base de internacional (Kaggle)
├── graphs/
│   ├── encargos_por_fumante.png
│   ├── encargos_por_regiao.png
│   ├── encargos_por_sexo.png
│   ├── imc_vs_encargos_fumante.png
│   ├── r2_comparacao_modelos.png
│   ├── real_vs_previsto_dt.png
│   ├── real_vs_previsto_lr.png
│   └── real_vs_previsto_rf.png
├── results/
│   └── results_modelos.csv             # Resultados comparativos dos modelos
├── database_generator.py               # Código para gerar ou manipular a base
├── main.py                             # Código principal de execução dos modelos
├── requirements.txt                    # Dependências do projeto
└── README.md
```

## Como Executar

1. Clone o repositório:

```bash
git clone git@github.com:tetyemillynha/health-plan-forecast.git
cd health-plan-forecast
```

2. Crie um ambiente virtual (opcional, mas recomendado):

```bash
python -m venv venv
source venv/bin/activate  # (Linux/Mac)
venv\Scripts\activate     # (Windows)
```

3. Instale as dependências:

```bash
pip install -r requirements.txt
```

4. Execute o projeto:

```bash
python main.py
```

Os gráficos gerados serão salvos na pasta `graphs/` e os resultados comparativos na pasta `results/`.

## Tecnologias Utilizadas

* Python
* Pandas
* Scikit-learn
* Matplotlib
* Seaborn

## Autor

* \[Seu Nome]
* Projeto acadêmico de Machine Learning

---

Se quiser, eu posso agora gerar o arquivo `README.md` já prontinho pra você só copiar e colar no seu projeto — inclusive adaptando o link do repositório e o seu nome, se me informar.

Quer? 🚀
