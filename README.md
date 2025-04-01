# 📊 Previsão de Vendas com Machine Learning 🤖

Este projeto tem como objetivo desenvolver um modelo de **Machine Learning** para prever as vendas futuras de um atacado, utilizando dados históricos de vendas. Com essa previsão, a empresa pode otimizar a gestão de estoque, evitar perdas e aumentar a eficiência nas estratégias de venda.

## 🛠 Tecnologias Utilizadas

- 🐍 **Python**
- 📊 **Pandas** para manipulação e análise de dados
- 📋 Pandas Profiling para geração de relatórios exploratórios automáticos
- 🤖 **Scikit-Learn** para construção e avaliação dos modelos de Machine Learning

## 🔍 Etapas do Projeto

1. **📂 Coleta e Limpeza dos Dados**: Foram utilizados dados de vendas dos últimos três anos, com informações sobre produtos e lojas.
2. **📊 Análise Exploratória**: Visualização dos padrões de vendas e identificação de tendências.
3. **⚙️ Construção do Modelo**:
   - Teste de diferentes algoritmos, como **Regressão Linear** e **Random Forest Regressor**.
   - Avaliação de desempenho com métricas como **RMSE (Root Mean Squared Error)**.
4. **🛠 Validação e Ajuste**: Refinamento do modelo para melhorar a precisão.
5. **📈 Geração das Previsões**: Predição das vendas futuras para otimização do estoque e estratégias comerciais.

## 💻 Código do Projeto

```python
# 📚 Importação das Bibliotecas
import pandas as pd
from pandas_profiling import ProfileReport
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# 📥 Importando os dados
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# 🔍 Visualizar os dez primeiros registros
train_data.head(10)

# 🔄 Separando variáveis de entrada e saída
train_x = train_data.drop(columns=['Item_Outlet_Sales'], axis=1)
train_y = train_data['Item_Outlet_Sales']
test_x = test_data.drop(columns=['Item_Outlet_Sales'], axis=1)
test_y = test_data['Item_Outlet_Sales']

# 🔢 Criando o modelo de Regressão Linear
model_L = LinearRegression()
model_L.fit(train_x, train_y)

# 🌳 Criando o modelo Random Forest Regressor
model_RFR = RandomForestRegressor(max_depth=5)
model_RFR.fit(train_x, train_y)

# 📈 Fazendo previsões
predict_train = model_RFR.predict(train_x)
predict_test = model_RFR.predict(test_x)

# 📉 Calculando o RMSE
rmse_train = mean_squared_error(train_y, predict_train)**(0.5)
rmse_test = mean_squared_error(test_y, predict_test)**(0.5)

print(f'📊 RMSE Treino: {rmse_train}')
print(f'📊 RMSE Teste: {rmse_test}')
```

## 📌 Resultados e Conclusão

O modelo final foi capaz de identificar padrões relevantes e fornecer previsões confiáveis. A empresa pode agora utilizar essas previsões para **evitar excesso ou falta de produtos, otimizar a distribuição e melhorar a eficiência operacional**. Futuramente, melhorias podem incluir a adição de novas variáveis e ajustes no modelo para aumentar ainda mais a precisão.



