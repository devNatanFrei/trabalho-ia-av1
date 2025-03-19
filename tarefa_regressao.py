import numpy as np
import matplotlib.pyplot as plt


# Tarefa de Regressão.
data = np.loadtxt('atividade_enzimatica.csv', delimiter=',')

#  1. Faça uma visualização inicial dos dados através do gráfico de espalhamento. 
#  Nessa etapa, faça discussões sobre quais serão as características de um modelo que 
#  consegue entender o padrão entre variáveis regressoras e variáveis observadas.

temp = data[:, 0]
ph = data[:, 1]
ativ_enz = data[:, 2]

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(temp, ativ_enz, color='red', marker='x')
plt.xlabel('Temperatura (°C)')
plt.ylabel('Atividade Enzimática (U/mg)')
plt.title('Temperatura x Atividade Enzimática')

plt.subplot(1, 2, 2)
plt.scatter(ph, ativ_enz, color='blue', marker='x')
plt.xlabel('pH')
plt.ylabel('Atividade Enzimática (U/mg)')
plt.title('pH x Atividade Enzimática')


# plt.show()


# 2. Em seguida, organize os dados de modo que as vari´aveis regressoras sejam armazenadas em uma matriz
# (X) de dimens˜ao R
# N×p
# . Fa¸ca o mesmo para o vetor de vari´avel dependente (y), organizando em um
# vetor de dimens˜ao R
# N×1
# .
X = data[:, :2]
y = ativ_enz

def print_matrix(matrix, name):
    print(f"Matriz {name}:\n")
    for row in matrix:
        print(row)
        
print_matrix(X, "X (N×p)")
print_matrix(y, "y (N×1)")

# 3. Os modelos a serem implementados nessa etapa ser˜ao: MQO tradicional, MQO regularizado (Tikhonov) e M´edia de valores observ´aveis.Obs: lembre-se que todos os modelos tamb´em estimam o valor
# do intercepto.

# MQO Tradicional (OLS - Mínimos Quadrados Ordinários)
N = X.shape[0]
X = np.hstack((np.ones((N, 1)), X))

mqo_trad = np.linalg.inv(X.T @ X) @ X.T @ y
print(f"MQO Tradicional (OLS): {mqo_trad}")

# MQO Regularizado (Tikhonov/Ridge) com lambda = 0.1
lambda_ = 0.1
I = np.eye(X.shape[1])
mqo_req = np.linalg.inv(X.T @ X + lambda_ * I) @ X.T @ y
print(f"MQO Regularizado (Tikhonov/Ridge): {mqo_req}") 

# Média dos valores observáveis
media_y = np.mean(y)
print(f"Média dos valores observáveis: {media_y}")

# 4. Para o modelo regularizado, h´a a dependˆencia da defini¸c˜ao de seu hiperparˆametro λ. Assim, sua equipe
# deve testar o presente modelo para os seguintes valores de lambda:
# λ = {0, 0.25, 0.5, 0.75, 1}
# . Assim, ao todo, existir˜ao 6 estimativas diferentes do vetor β ∈ R
# p+1×1

lambdas = [0, 0.25, 0.5, 0.75, 1]
betas = {}

for lambda_ in lambdas:
    beta = np.linalg.inv(X.T @ X + lambda_ * I) @ X.T @ y
    betas[lambda_] = beta
    
    
print("Estimativas do vetor β para diferentes valores de λ:")
for lambda_, beta in betas.items():
    print(f"λ = {lambda_}: {beta}")
    
    
# 5. Para validar os modelos utilizados na tarefa de regress˜ao, sua equipe deve projetar a valida¸c˜ao utilizando
# as simula¸c˜oes por Monte Carlo. Nessa etapa, defina a quantidade de rodadas da simula¸c˜ao igual a
# R = 500. Em cada rodada, deve-se realizar o particionamento em 80% dos dados para treinamento e
# 20% para teste. A medida de desempenho de cada um dos 5 modelos diferentes deve ser a soma dos
# desvios quadr´aticos (RSS - Residual Sum of Squares) e cada medida obtida deve ser armazenada em uma
# lista.
    
R = 500





