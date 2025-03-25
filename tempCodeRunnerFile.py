import numpy as np
import matplotlib.pyplot as plt

# Carregando os dados
data = np.loadtxt('atividade_enzimatica.csv', delimiter=',')

temp = data[:, 0]
ph = data[:, 1]
ativ_enz = data[:, 2]

fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(temp, ph, ativ_enz, c='r', marker='x')
ax.set_xlabel('Temperatura')
ax.set_ylabel('pH')
ax.set_zlabel('Atividade Enzimática')

X = data[:, :2]
y = ativ_enz.reshape(-1, 1)     

N = X.shape[0]
X = np.hstack((np.ones((N, 1)), X))

beta_mqo = np.linalg.inv(X.T @ X) @ X.T @ y
print(f"MQO Tradicional (OLS): {beta_mqo}")


lambdas = [0, 0.25, 0.5, 0.75, 1]  
betas = {}

def ridge_regression(X, y, lambda_):
    I = np.eye(X.shape[1])  
    I[0, 0] = 0  
    beta = np.linalg.inv(X.T @ X + lambda_ * I) @ X.T @ y
    return beta

for lambda_ in lambdas:
    betas[lambda_] = ridge_regression(X, y, lambda_)

print()
print("Estimativas do vetor β para diferentes valores de λ: ")
for lambda_, beta in betas.items():
    print(f"λ = {lambda_}: {beta}")

print()
media_y = np.mean(y)
print(f"Média dos valores observáveis: {media_y}")

    
# Questão 5 e 6
media_mqo_tradicional = np.mean(beta_mqo)
print(f"Média dos valores estimados pelo MQO tradicional: {media_mqo_tradicional}")

desvio_padrao_mqo = np.std(beta_mqo)
print(f"Desvio padrão dos valores estimados pelo MQO tradicional: {desvio_padrao_mqo}")

valor_maximo_mqo = np.max(beta_mqo)
valor_minimo_mqo = np.min(beta_mqo)
print(f"Valor máximo estimado pelo MQO tradicional: {valor_maximo_mqo}")
print(f"Valor mínimo estimado pelo MQO tradicional: {valor_minimo_mqo}")

print()



rss_results = {lambda_: [] for lambda_ in lambdas} 
rss_y = []

for _ in range(500):
   
    indices = np.random.permutation(len(y))
    split_idx = int(0.8 * len(y))  
    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]
    
 
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    
    for lambda_ in lambdas:
        beta = ridge_regression(X_train, y_train, lambda_)
        y_pred = X_test @ beta  
        rss = np.sum((y_test - y_pred) ** 2) 
        rss_results[lambda_].append(rss)
    
    y_pred_media = np.full(y_test.shape, np.mean(y_train))
    rss_y.append(np.sum((y_test - y_pred_media) ** 2))


stats = {}
for lambda_ in lambdas:
    rss_list = rss_results[lambda_]
    stats[lambda_] = {
        'mean': np.mean(rss_list),
        'std': np.std(rss_list),
        'max': np.max(rss_list),
        'min': np.min(rss_list)
    }

rss_y_media = np.mean(rss_y)
rss_y_desvio = np.std(rss_y)
rss_y_max = np.max(rss_y)
rss_y_min = np.min(rss_y)
 
print(f"Média do RSS para y médio: {rss_y_media}")
print(f"Desvio padrão do RSS para y médio: {rss_y_desvio}")
print(f"Valor máximo do RSS para y médio: {rss_y_max}")
print(f"Valor mínimo do RSS para y médio: {rss_y_min}")
print()

print("λ | Mean RSS | Std RSS | Max RSS | Min RSS")
print("-" * 50)
for lambda_ in lambdas:
    print(f"{lambda_:.2f} | {stats[lambda_]['mean']:.6f} | {stats[lambda_]['std']:.6f} | "
          f"{stats[lambda_]['max']:.4f} | {stats[lambda_]['min']:.4f}")


plt.show()