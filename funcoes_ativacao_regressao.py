import numpy as np

# Função step
def step_function(soma):
    if soma >=1:
        return 1
    return 0

print(step_function(5.2)) # caso maior ou igual a 1, resultado 1. Caso menor, resultado 0
print(step_function(2.5))
print(step_function(1.1))
# função sigmoid - Função de probabilidade. Exemplo: Gato = 1 e Cachorro = 0. Quanto mais proximo de 1, mais chances de ser gato
# quanto mais proximo de 0, mais chances de ser cachorro.
def sigmoid(soma):
    return 1 / (1 + np.exp(-soma)) # 1 / (1 + np.exp(-soma)) - Isso é uma formula da função 

print(f'valor sigmoid {sigmoid(5)}')
print(f'valor sigmoid {sigmoid(2)}')
print(f'valor sigmoid {sigmoid(1)}')
print(f'valor sigmoid {sigmoid(-2)}')


# Função tangente hiperbólica (tahn) Mapeia os valores de entrada para o intervalo de -1 a 1, 
# o que permite lidar melhor com valores negativos e centralizar os dados em torno de zero.
def tahn_function(soma):
    return (np.exp(soma) - np.exp(-soma)) / (np.exp(soma) + np.exp(-soma)) 

print(f'valor tahn {tahn_function(-10)}') # valor tahn -0.9999999958776926
print(f'valor tahn {tahn_function(10)}') # valor tahn 0.9999999958776926
print(f'valor tahn {tahn_function(1)}') # valor tahn 0.7615941559557649

# ReLU - realiza o processamento dos dados dentro das camadas ocultas, seu valor parte de 0 para cima. Transforma qualquer valor 
# negativo em 0
def relu_function(soma):
    if soma >=0:
        return soma
    return 0

print(relu_function(-20)) # resposta: 0
print(relu_function(5)) # resposta: 5


# Função softmax - parecido com sigmoide, mas sigmoid atende somente duas classes (ex: gato e cachorro), a softmax é para mais classes
def softmax_function(x):
    ex = np.exp(x)
    return ex / ex.sum() 

print(softmax_function([7.0, 2.0, 1.3])) # Cada número é como se fosse uma classe. Ex: gato, cachorro, coelho
# Resposta [0.99001676 0.00667068 0.00331256]


# Função Linear - Usada para regressão com números, por exemplo, adivinhar gastos futuros de cartão de crédito 
# baseado no historico de gasto. Simples aplicação.

def linar_function(soma):
    return soma

print(linar_function(-10))
print(linar_function(10))
print(linar_function(1))


