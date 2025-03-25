import numpy as np
import matplotlib.pyplot as plt


dados = np.loadtxt('atividade_enzimatica.csv', delimiter=',')

temperatura = dados[:, 0]
ph = dados[:, 1]
atividade_enzimatica = dados[:, 2]

fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(temperatura, ph, atividade_enzimatica, c='r', marker='x')
ax.set_xlabel('Temperatura')
ax.set_ylabel('pH')
ax.set_zlabel('Atividade Enzimática')

X = dados[:, :2]
y = atividade_enzimatica.reshape(-1, 1)     

N = X.shape[0]
X = np.hstack((np.ones((N, 1)), X))

beta_mqo = np.linalg.inv(X.T @ X) @ X.T @ y
print(f"MQO Tradicional (OLS): {beta_mqo}")

lambdas = [0, 0.25, 0.5, 0.75, 1]  
betas = {}

def regressao_ridge(X, y, lambda_):
    I = np.eye(X.shape[1])  
    I[0, 0] = 0  
    beta = np.linalg.inv(X.T @ X + lambda_ * I) @ X.T @ y
    return beta

for lambda_ in lambdas:
    betas[lambda_] = regressao_ridge(X, y, lambda_)

print()
print("Estimativas do vetor β para diferentes valores de λ: ")
for lambda_, beta in betas.items():
    print(f"λ = {lambda_}: {beta}")

print()
media_y = np.mean(y)
print(f"Média dos valores observáveis: {media_y}")

# Questão 5 e 6
rss_resultados = {lambda_: [] for lambda_ in lambdas} 
rss_y = []

for _ in range(500):
    indices = np.random.permutation(len(y))
    indice_divisao = int(0.8 * len(y))  
    indices_treino = indices[:indice_divisao]
    indices_teste = indices[indice_divisao:]
    
    X_treino, X_teste = X[indices_treino], X[indices_teste]
    y_treino, y_teste = y[indices_treino], y[indices_teste]
    
    for lambda_ in lambdas:
        beta = regressao_ridge(X_treino, y_treino, lambda_)
        y_pred = X_teste @ beta  
        rss = np.sum((y_teste - y_pred) ** 2) 
        rss_resultados[lambda_].append(rss)
    
    y_pred_media = np.full(y_teste.shape, np.mean(y_treino))
    rss_y.append(np.sum((y_teste - y_pred_media) ** 2))

estatisticas = {}
for lambda_ in lambdas:
    rss_lista = rss_resultados[lambda_]
    estatisticas[lambda_] = {
        'media': np.mean(rss_lista),
        'desvio_padrao': np.std(rss_lista),
        'maximo': np.max(rss_lista),
        'minimo': np.min(rss_lista)
    }

rss_y_media = np.mean(rss_y)
rss_y_desvio = np.std(rss_y)
rss_y_maximo = np.max(rss_y)
rss_y_minimo = np.min(rss_y)
 
print(f"Média do RSS para y médio: {rss_y_media}")
print(f"Desvio padrão do RSS para y médio: {rss_y_desvio}")
print(f"Valor máximo do RSS para y médio: {rss_y_maximo}")
print(f"Valor mínimo do RSS para y médio: {rss_y_minimo}")
print()

print("λ | Média RSS | Desvio Padrão RSS | Máximo RSS | Mínimo RSS")
print("-" * 50)
for lambda_ in lambdas:
    print(f"{lambda_:.2f} | {estatisticas[lambda_]['media']:.6f} | {estatisticas[lambda_]['desvio_padrao']:.6f} | "
          f"{estatisticas[lambda_]['maximo']:.4f} | {estatisticas[lambda_]['minimo']:.4f}")

plt.show()
