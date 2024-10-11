import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split

# import train_test_split - faz a divisão da base de dadps entre treino e teste

# Aqui vamos utilizar dois arquivos csv, o entradas_breast e saidas_breast - Nester arquivos temos dados
# como 0=tumor benigno e 1= tumar maligno. Ensinar o algoritmo a aprender e classificar se o tumor é maligno ou benigno.

# Atributos previsores - fazer previsão
# é regra neste caso criar uma variável utlizando X
x = pd.read_csv('D:\ia_python\ia_cod\entradas_breast.csv')

# classe - na classe, utilizamos Y
y = pd.read_csv('D:\ia_python\ia_cod\saidas_breast.csv')

print(x)
print(y)

# Resposta: 
# 565  0 - linha 565, tumor benigno
# 566  0 - linha 566, benigno
# 567  0 - linha 567, benigno
# 568  1 - linha 568, maligno
# ENTENDENDO A RESPOSTA ACIMA: Juntando todos as colunas do arquivos entradas.csv, que tem informações sobre 
# os tumores(tamanho, raio, frequencia, etc), o programa consegue verificar se o tumor é maligno ou benigno.


"""Depois de reconhecer o padrão, vamos utilizar uma parte desses dados para treinar a rede neural e outra parte para fazer
testes depois que a rede neural estiver treinada e vamos instalar outra biblioteca chamada skleran - a mais famosa pra aprendizado
de maquina no python - install scikit-learn"""

x_treinamento, x_teste, y_treinamento, y_teste =  train_test_split(x, y, test_size=0.25)
# Como parâmetro x e y pois são as bases de dados completa. test_size=0.25 indica que 25% dos dados serão utilizados para testar e 
# o restante, ou seja, 75%, serão usados para treinar

print(f'Total de {x_treinamento.shape} registros para treinar(75%)')
# Resposta: Total de (426, 30) registros para treinar(75%) - o número 30 representa as características dos tumores
print(f'Total de {x_teste.shape} registros para testar (25%)')
# Resposta: Total de (143, 30) registros para testar (25%)

print(f'Total de {y_treinamento.shape} registros para treinar(75%)')