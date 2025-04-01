# Importação das Bibliotecas necessárias para o trabalho
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import  RandomForestRegressor
from sklearn.model_selection import train_test_split


# Instalando pandas-profiling
!pip install -U pandas-profiling

# Estrutura dos Dados
from PIL import Image
%matplotlib inline
im = Image.open("PV.png")
im.show()
im

# Importando os dados de treino e teste
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Visualizar os 10 primeiros registros
train_data.head(10)

# shape dos dados
print('\nShape of training data :',train_data.shape)
print('\nShape of testing data :',test_data.shape)


from pandas_profiling import ProfileReport
profile = ProfileReport(train_data, title='Vendas do Supermercado Comercial Esperanca',html={'style':{'full_width':True}})
profile.to_notebook_iframe()
profile.to_file(output_file="Relatório- Base de Dados.html")

# Pré-processamento dos dados - Amostra Treino e Teste

# Dados de Treino - Preparação para modelagem - Separação das  variáveis de entrada e variável de saida (target/label.....o que queremos prever...)
train_x = train_data.drop(columns=['Item_Outlet_Sales'],axis=1)
train_y = train_data['Item_Outlet_Sales']

# Dados de Teste - Preparação para modelagem - Separação das  variáveis de entrada e variável de saida (target/label.....o que queremos prever...)
test_x = test_data.drop(columns=['Item_Outlet_Sales'],axis=1)
test_y = test_data['Item_Outlet_Sales']

# Modelagem

## Criação do Modelo - Linear Regression
model_L = LinearRegression()

# obs: Você pode adicionar parâmetros e testar para ver se melhora o resultado da sua predição
# Como por exemplo os parâmetros "fit_intercept" e "normalize"
# Documentação do sklearn LinearRegression:

# Treino do modelo - Dados de Treino
model_L.fit(train_x,train_y)

# Coeficientes do modelo Treinado
print('\nCoefficient of model :', model_L.coef_)

# intercepto do Modelo
print('\nIntercept of model',model_L.intercept_)

# Fazendo Previsões com os dados de treino
predict_train = model_L.predict(train_x)
print('\nItem_Outlet_Sales on training data',predict_train)

# Calculando o RMSE Root Mean Squared Error - nos dados de treino
rmse_train = mean_squared_error(train_y,predict_train)**(0.5)
print('\nRMSE on train dataset : ', rmse_train)

# Fazendo Previsões com os dados de teste
predict_test = model_L.predict(test_x)
print('\nItem_Outlet_Sales on test data',predict_test)


# Calculando o RMSE Root Mean Squared Error - nos dados de teste
rmse_test = mean_squared_error(test_y,predict_test)**(0.5)
print('\nRMSE on test dataset : ', rmse_test)

# Criando o Modelo Mais Top Top - RandomForestRegressor
model_RFR = RandomForestRegressor(max_depth=5)

# Treinando o modelo - Dados de Treino
model_RFR.fit(train_x, train_y)

# Fazendo previsões com os dados de treino e teste
predict_train = model_RFR.predict(train_x)
predict_test = model_RFR.predict(test_x)

# Calculando o RMSE Root Mean Squared Error
print('RMSE on train data: ', mean_squared_error(train_y, predict_train)**(0.5))
print('RMSE on test data: ',  mean_squared_error(test_y, predict_test)**(0.5))

# Fazendo novas previsões predict the target on the testing dataset
predict_test = model_RFR.predict(test_x)
print('\nItem_Outlet_Sales on test data',predict_test)

